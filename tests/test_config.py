import yaml

from spine.config import SpineConfig, from_yaml


def test_default_config_builds() -> None:
    cfg = SpineConfig()
    assert cfg.model.sample_rate == 24000
    assert cfg.model.hop_length == 512
    assert cfg.loss.lambda_mel == 15.0
    assert cfg.optimizer.betas == (0.8, 0.9)
    assert cfg.training.batch_size == 32


def test_from_yaml_sparse_override(tmp_path) -> None:
    path = tmp_path / "override.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"transformer_dim": 128, "encoder_rates": [2, 2]},
                "loss": {"lambda_mel": 42.0},
            }
        )
    )
    cfg = from_yaml(path)

    assert cfg.model.transformer_dim == 128
    assert cfg.model.encoder_rates == (2, 2)
    assert cfg.loss.lambda_mel == 42.0
    assert cfg.model.sample_rate == 24000
    assert cfg.optimizer.lr == 1e-4
