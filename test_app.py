import unittest,tempfile,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import test_core
import joblib
from streamlit.testing.v1 import AppTest

class AppTests(unittest.TestCase):
    def test_ui_single_history_evaluation(self):
        test_core.CoreTests.setUpClass()
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for name in ['app.py','core.py']:shutil.copy(root/name,p/name)
            bundle={**test_core.CoreTests.bundle,'model_name':'TEST FIXTURE','metadata':test_core.CoreTests.meta}
            joblib.dump(bundle,p/'mo_hinh_demo.joblib')
            app=AppTest.from_file(str(p/'app.py')).run(timeout=30)
            self.assertEqual(len(app.exception),0)
            app.button[0].click().run(timeout=30)
            self.assertEqual(len(app.exception),0)
            self.assertEqual(len(app.session_state['history']),1)
            app.sidebar.radio[0].set_value('Lịch sử').run()
            self.assertEqual(len(app.exception),0)
            app.button[0].click().run()
            self.assertEqual(len(app.session_state['history']),0)
            app.sidebar.radio[0].set_value('CSV hàng loạt').run()
            self.assertEqual(len(app.exception),0)
            app.sidebar.radio[0].set_value('Đánh giá').run()
            self.assertEqual(len(app.exception),0)
            app.sidebar.radio[0].set_value('Thông tin').run()
            self.assertEqual(len(app.exception),0)
if __name__=='__main__':unittest.main()
