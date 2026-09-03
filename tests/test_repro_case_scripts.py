import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


class ReproCaseScriptsTests(unittest.TestCase):
    def test_generate_repro_case_outputs_files(self):
        subprocess.check_call(["python3", str(REPO / "scripts" / "generate_repro_case.py")])

        for case_name in ("quick", "reference"):
            case_dir = REPO / "cases" / "cutting2d" / case_name
            self.assertTrue((case_dir / "ProjectParameters.json").exists())
            self.assertTrue((case_dir / "Materials.json").exists())

            with (case_dir / "ProjectParameters.json").open(encoding="utf-8") as f:
                project = json.load(f)

            self.assertEqual(project["problem_data"]["domain_size"], 2)
            self.assertEqual(project["problem_data"]["analysis_type"], "quasi_static")

    def test_postprocess_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            forces = tmp_path / "forces.csv"
            energies = tmp_path / "energies.csv"
            chip = tmp_path / "chip.csv"
            out = tmp_path / "summary.json"

            forces.write_text(
                "time,cutting_force,thrust_force,resultant_force,apparent_mu\n"
                "0.0,10,2,10.198,0.2\n"
                "0.2,20,3,20.223,0.15\n"
                "0.3,24,4,24.331,0.16\n",
                encoding="utf-8",
            )
            energies.write_text(
                "time,internal_energy,kinetic_energy,external_work,contact_dissipation,plastic_dissipation\n"
                "0.1,1,0.2,1.2,0.1,0.4\n"
                "0.3,3,0.4,3.2,0.3,1.1\n",
                encoding="utf-8",
            )
            chip.write_text(
                "step,chip_thickness,undeformed_thickness\n"
                "1,0.00025,0.00020\n",
                encoding="utf-8",
            )

            subprocess.check_call(
                [
                    "python3",
                    str(REPO / "scripts" / "postprocess_cutting_metrics.py"),
                    "--forces",
                    str(forces),
                    "--energies",
                    str(energies),
                    "--chip",
                    str(chip),
                    "--transient-cutoff-time",
                    "0.15",
                    "--output",
                    str(out),
                ]
            )

            with out.open(encoding="utf-8") as f:
                summary = json.load(f)

            self.assertAlmostEqual(summary["mean_cutting_force"], 22.0)
            self.assertAlmostEqual(summary["chip_compression_ratio"], 1.25)


if __name__ == "__main__":
    unittest.main()
