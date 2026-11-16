"""Standard-library exporter regression tests. No simulator or third-party packages."""
import ast
import math
from pathlib import Path
import sys
import unittest

TOOLS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS))
from solo12_twofeet_pose_viewer import config_literal, task_defaults, build_html as pose_html
from solo12_joint_web_viewer import build_html


class ExportTests(unittest.TestCase):
    def test_source_defaults_include_annotations_and_math_degrees(self):
        data = task_defaults()
        self.assertEqual(len(data['joint_names']), 12)
        self.assertEqual(len(data['joint_pose']), 12)
        self.assertEqual(data['joint_physical_limit_left_hip'], [-math.degrees(1.22), math.degrees(2.35)])
        self.assertTrue(data['use_asymmetric_hip_limits'])
        self.assertFalse(data['use_asymmetric_thigh_limits'])
        self.assertIsNone(data['joint_soft_limit_front_thigh'])
        for joint in ('hip', 'thigh', 'calf'):
            self.assertIn('joint_soft_limit_'+joint+'_delta', data)

    def test_only_allowlisted_expressions_can_be_read(self):
        for text in ('__import__("os").getcwd()', 'math.sin(1)', 'math.degrees(1+1)', 'math.degrees(x=1)'):
            with self.subTest(text=text), self.assertRaises((ValueError, TypeError)):
                config_literal(ast.parse(text, mode='eval').body)
        self.assertEqual(config_literal(ast.parse('(-math.degrees(1), math.degrees(2))', mode='eval').body), [-math.degrees(1), math.degrees(2)])

    def test_both_viewers_export_without_simulator_or_external_assets(self):
        for builder in (build_html, pose_html):
            with self.subTest(builder=builder.__module__):
                html = builder()
                self.assertIn('FL_hip_joint', html)
                self.assertNotIn('/* MODEL_DATA */null', html)
                self.assertNotIn('/* THREE_LIBRARY */', html)
                self.assertNotIn('<script src=', html)
        self.assertIn('config_sha256', build_html())


if __name__ == '__main__':
    unittest.main()
