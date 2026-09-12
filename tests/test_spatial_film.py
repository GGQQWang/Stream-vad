import torch

from spatial_film import SpatialFiLM


def test_spatial_film_zero_init_is_identity_and_logs_stats():
    film = SpatialFiLM(d_ssm=4, llm_hidden=8, hidden=6)
    x = torch.randn(3, 5, 8)
    h = torch.randn(3, 4)
    modulated, stats = film(x, h)
    assert torch.allclose(modulated, x, atol=1e-6)
    assert stats["film_gamma_abs_mean"] == 0.0
    assert stats["film_beta_abs_mean"] == 0.0
    assert stats["film_delta_rel"] == 0.0


def test_spatial_film_supports_bfloat16_inputs_to_float32_module():
    film = SpatialFiLM(d_ssm=4, llm_hidden=8, hidden=6)
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    h = torch.randn(2, 4, dtype=torch.bfloat16)
    modulated, _ = film(x, h)
    assert modulated.dtype == torch.float32
    assert modulated.shape == (2, 3, 8)
