import subprocess


def test_training() -> None:
    """Test training script via CLI invocation."""
    
    process = subprocess.Popen([
            "python", "scripts/cli/training.py",
            "working_dir=runs",
            "job_name=train",
            "data=ir/patches",
            "data_path=tests/test_data/ir_dataset",
            "data.IR.preprocessor_arguments.patch_size=125",
            "data.Formula.column=molecular_formula",
            "model=custom_model",
            "molecules=True",
            "trainer.epochs=1"
        ],
        shell=False)
    process.communicate()
    
    assert process.returncode == 0


def test_predicting() -> None:
    """Test predict script via CLI invocation."""
    
    process = subprocess.Popen([
            "python", "scripts/cli/predict.py",
            "working_dir=runs",
            "job_name=predict",
            "data=ir/patches",
            "data_path=tests/test_data/ir_dataset",
            "data.IR.preprocessor_arguments.patch_size=125",
            "data.Formula.column=molecular_formula",
            "model=custom_model",
            "molecules=True",
            "preprocessor_path=runs/train/preprocessor.pkl",
            "model.model_checkpoint_path=runs/train/version_0/checkpoints/last.ckpt"
        ],
        shell=False)
    process.communicate()
    
    assert process.returncode == 0
