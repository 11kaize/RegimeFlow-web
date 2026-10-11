"""API wiring/failure contracts; run with backend requirements installed."""
import json
import unittest
from unittest.mock import patch
import numpy as np
try:
    from fastapi.testclient import TestClient
    from web.backend import server
except ImportError as exc:
    raise unittest.SkipTest('Backend API dependencies unavailable: '+str(exc))


class APIContracts(unittest.TestCase):
    def setUp(self):
        self.client=TestClient(server.app)
        self.loader=patch.object(server,'_load_regimeflow_engine',return_value=None)
        self.loader.start();self.addCleanup(self.loader.stop)

    def test_all_routes_fail_when_model_missing(self):
        with patch.object(server,'regimeflow_engine',None):
            for path,body in [('/api/predict',{'context':[1,2,3,4]}),('/api/predict/multi',{'contexts':[[1,2,3,4]]}),('/api/predict/multi/stream',{'contexts':[[1,2,3,4]]})]:
                response=self.client.post(path,json=body)
                self.assertEqual(response.status_code,503)
                self.assertNotIn('predictions',response.text)

    def test_batch_and_stream_keep_independent_species_arguments(self):
        calls=[]
        class Engine:
            def predict(self,**args):
                calls.append(args); return np.arange(256,dtype=float)
        body={'contexts':[[1,2,3,4],[5,6,7,8]],'traj_patterns':[3,5],'periods':[20,0],'bounds':[[0,4],[0,8]],'prediction_length':5}
        with patch.object(server,'regimeflow_engine',Engine()):
            for path in ['/api/predict/multi','/api/predict/multi/stream']:
                response=self.client.post(path,json=body)
                self.assertEqual(response.status_code,200)
            self.assertEqual([call['traj_pattern'] for call in calls],[3,5,3,5])
            self.assertEqual([call['period'] for call in calls],[20,0,20,0])
            self.assertEqual(calls[1]['bounds'],[0,8])
            for line in response.text.splitlines():
                result=json.loads(line)['result'];self.assertEqual(result['model'],'regimeflow');self.assertEqual(len(result['predictions']),5)
        mismatch=self.client.post('/api/predict/multi',json={'contexts':[[1,2]],'periods':[1,2]})
        self.assertEqual(mismatch.status_code,400)

    def test_prediction_exception_never_becomes_a_synthetic_success(self):
        class Engine:
            def predict(self,**args): raise RuntimeError('injected failure')
        with patch.object(server,'regimeflow_engine',Engine()):
            response=self.client.post('/api/predict',json={'context':[1,2,3,4]})
            self.assertEqual(response.status_code,500)
            response=self.client.post('/api/predict/multi/stream',json={'contexts':[[1,2,3,4]]})
            line=json.loads(response.text)
            self.assertIn('error',line);self.assertNotIn('result',line)

    def test_stream_lines_carry_ok_discriminator(self):
        class Engine:
            def predict(self,**args):
                if args['context'][0]==0: raise RuntimeError('injected failure')
                return np.arange(256,dtype=float)
        with patch.object(server,'regimeflow_engine',Engine()):
            response=self.client.post('/api/predict/multi/stream',json={'contexts':[[1,2,3,4],[0,2,3,4]]})
            ok_line,failed_line=[json.loads(line) for line in response.text.splitlines()]
            self.assertIs(ok_line['ok'],True);self.assertIn('result',ok_line);self.assertNotIn('error',ok_line)
            self.assertIs(failed_line['ok'],False);self.assertIn('error',failed_line);self.assertNotIn('result',failed_line)

    def test_out_of_range_traj_patterns_are_rejected(self):
        for body in [{'contexts':[[1,2,3,4]],'traj_patterns':[99]},
                     {'contexts':[[1,2,3,4]],'traj_patterns':[-5]},
                     {'contexts':[[1,2,3,4],[5,6,7,8]],'traj_patterns':[3,99]}]:
            self.assertEqual(self.client.post('/api/predict/multi',json=body).status_code,422)
            self.assertEqual(self.client.post('/api/predict/multi/stream',json=body).status_code,422)

    def test_proxy_uses_distinct_csv_and_conditions_sources(self):
        record={'X':{'trajectory_type':3,'period':25}}
        def fetch(mid,name,suffix): return json.dumps(record).encode() if suffix=='_conditions.json' else b'time,X\n0,1\n'
        with patch.object(server,'_fetch_dataset_file',side_effect=fetch) as fetcher:
            self.assertEqual(self.client.get('/api/conditions/ID/Name').json(),record)
            self.assertIn('time,X',self.client.get('/api/csv/ID/Name').text)
            self.assertEqual([call.args[-1] for call in fetcher.call_args_list],['_conditions.json','.csv'])


if __name__=='__main__': unittest.main()
