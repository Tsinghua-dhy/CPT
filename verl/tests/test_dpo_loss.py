"""
Unit tests for offline DPO loss and log probability computation.
Run with: python -m pytest tests/test_dpo_loss.py -v
Or simply: python tests/test_dpo_loss.py
"""

import math
import sys
import torch
import torch.nn.functional as F

# Add verl to path
# sys.path.insert(0, "<repo-root>")

from verl.trainer.fsdp_dpo_trainer import compute_log_probs, dpo_loss


class TestComputeLogProbs:
    """Tests for compute_log_probs function."""

    def test_basic_correctness(self):
        """Verify log probs are computed correctly on a simple example."""
        torch.manual_seed(42)
        batch_size, seq_len, vocab_size = 2, 8, 10

        logits = torch.randn(batch_size, seq_len, vocab_size)
        # labels = input_ids (the function does the shift internally)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        # loss_mask: only care about tokens 3-6 (response region)
        loss_mask = torch.zeros(batch_size, seq_len)
        loss_mask[:, 3:7] = 1.0

        result = compute_log_probs(logits, labels, loss_mask)

        assert result.shape == (batch_size,), f"Expected shape ({batch_size},), got {result.shape}"
        # Log probs should be negative (log of probability < 1)
        assert (result <= 0).all(), f"Log probs should be <= 0, got {result}"
        print(f"  [PASS] basic_correctness: result={result.tolist()}")

    def test_manual_computation(self):
        """Compare against manual per-token log prob calculation."""
        torch.manual_seed(123)
        batch_size, seq_len, vocab_size = 1, 5, 4

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[0, 1, 2, 3, 1]])
        loss_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])

        result = compute_log_probs(logits, labels, loss_mask)

        # Manual: shift logits[:-1] predicts labels[1:]
        shift_logits = logits[0, :-1, :]  # (4, 4)
        shift_labels = labels[0, 1:]  # (4,)
        shift_mask = loss_mask[0, :-1]  # (4,)

        log_probs_manual = torch.zeros(4)
        for t in range(4):
            log_softmax = F.log_softmax(shift_logits[t], dim=-1)
            log_probs_manual[t] = log_softmax[shift_labels[t]]

        expected = (log_probs_manual * shift_mask).sum()

        assert torch.allclose(result[0], expected, atol=1e-5), (
            f"Mismatch: got {result[0].item()}, expected {expected.item()}"
        )
        print(f"  [PASS] manual_computation: result={result[0].item():.6f}, expected={expected.item():.6f}")

    def test_zero_mask(self):
        """If loss_mask is all zeros, log probs should be zero."""
        torch.manual_seed(0)
        logits = torch.randn(2, 6, 10)
        labels = torch.randint(0, 10, (2, 6))
        loss_mask = torch.zeros(2, 6)

        result = compute_log_probs(logits, labels, loss_mask)
        assert (result == 0).all(), f"Expected all zeros, got {result}"
        print(f"  [PASS] zero_mask: result={result.tolist()}")


class TestDPOLoss:
    """Tests for dpo_loss function."""

    def test_chosen_preferred_loss_is_low(self):
        """When policy strongly prefers chosen (and ref is neutral), loss should be small."""
        policy_chosen = torch.tensor([-1.0])  # higher (less negative) log prob
        policy_rejected = torch.tensor([-5.0])
        ref_chosen = torch.tensor([-3.0])
        ref_rejected = torch.tensor([-3.0])  # ref is neutral

        loss, c_rew, r_rew = dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
        )

        # pi_logratio = -1 - (-5) = 4, ref_logratio = 0, logits = 4
        # loss = -log(sigmoid(0.1 * 4)) = -log(sigmoid(0.4)) ≈ 0.513
        expected_logits = 4.0
        expected_loss = -F.logsigmoid(torch.tensor(0.1 * expected_logits)).item()

        assert abs(loss.item() - expected_loss) < 1e-5, (
            f"Loss mismatch: got {loss.item():.6f}, expected {expected_loss:.6f}"
        )
        print(f"  [PASS] chosen_preferred: loss={loss.item():.6f}, expected={expected_loss:.6f}")

    def test_rejected_preferred_loss_is_high(self):
        """When policy prefers rejected, loss should be higher."""
        policy_chosen = torch.tensor([-5.0])
        policy_rejected = torch.tensor([-1.0])  # policy prefers rejected
        ref_chosen = torch.tensor([-3.0])
        ref_rejected = torch.tensor([-3.0])

        loss_bad, _, _ = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)

        # Now flip: policy prefers chosen
        loss_good, _, _ = dpo_loss(
            torch.tensor([-1.0]), torch.tensor([-5.0]),
            ref_chosen, ref_rejected, beta=0.1
        )

        assert loss_bad.item() > loss_good.item(), (
            f"Bad preference should have higher loss: {loss_bad.item()} vs {loss_good.item()}"
        )
        print(f"  [PASS] rejected_preferred: loss_bad={loss_bad.item():.4f} > loss_good={loss_good.item():.4f}")

    def test_reference_free(self):
        """reference_free should ignore ref logps."""
        policy_chosen = torch.tensor([-1.0])
        policy_rejected = torch.tensor([-3.0])
        ref_chosen = torch.tensor([-100.0])  # extreme values
        ref_rejected = torch.tensor([100.0])

        loss_rf, _, _ = dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected,
            beta=0.1, reference_free=True
        )

        # With reference_free, ref_logratios = 0, so logits = pi_logratios = 2
        expected = -F.logsigmoid(torch.tensor(0.1 * 2.0)).item()
        assert abs(loss_rf.item() - expected) < 1e-5, (
            f"reference_free loss mismatch: got {loss_rf.item():.6f}, expected {expected:.6f}"
        )
        print(f"  [PASS] reference_free: loss={loss_rf.item():.6f}")

    def test_ipo_loss(self):
        """Test IPO loss type."""
        policy_chosen = torch.tensor([-1.0])
        policy_rejected = torch.tensor([-3.0])
        ref_chosen = torch.tensor([-2.0])
        ref_rejected = torch.tensor([-2.0])
        beta = 0.5

        loss_ipo, _, _ = dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected,
            beta=beta, loss_type="ipo"
        )

        # logits = ((-1) - (-3)) - ((-2) - (-2)) = 2 - 0 = 2
        # ipo loss = (2 - 1/(2*0.5))^2 = (2 - 1)^2 = 1.0
        expected = (2.0 - 1.0 / (2.0 * beta)) ** 2
        assert abs(loss_ipo.item() - expected) < 1e-5, (
            f"IPO loss mismatch: got {loss_ipo.item():.6f}, expected {expected:.6f}"
        )
        print(f"  [PASS] ipo_loss: loss={loss_ipo.item():.6f}, expected={expected:.6f}")

    def test_reward_margin(self):
        """Reward margin = chosen_rewards - rejected_rewards should be positive when policy prefers chosen."""
        policy_chosen = torch.tensor([-1.0])
        policy_rejected = torch.tensor([-5.0])
        ref_chosen = torch.tensor([-3.0])
        ref_rejected = torch.tensor([-3.0])

        _, c_rew, r_rew = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)

        margin = c_rew.item() - r_rew.item()
        assert margin > 0, f"Reward margin should be positive when chosen preferred, got {margin}"

        # chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps) = 0.1 * (-1 - (-3)) = 0.2
        # rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps) = 0.1 * (-5 - (-3)) = -0.2
        assert abs(c_rew.item() - 0.2) < 1e-5
        assert abs(r_rew.item() - (-0.2)) < 1e-5
        print(f"  [PASS] reward_margin: chosen={c_rew.item():.4f}, rejected={r_rew.item():.4f}, margin={margin:.4f}")

    def test_label_smoothing(self):
        """Label smoothing should increase loss compared to no smoothing (for well-separated logits)."""
        policy_chosen = torch.tensor([-1.0])
        policy_rejected = torch.tensor([-5.0])
        ref_chosen = torch.tensor([-3.0])
        ref_rejected = torch.tensor([-3.0])

        loss_no_smooth, _, _ = dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected,
            beta=0.1, label_smoothing=0.0
        )
        loss_smooth, _, _ = dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected,
            beta=0.1, label_smoothing=0.1
        )

        assert loss_smooth.item() > loss_no_smooth.item(), (
            f"Smoothed loss should be higher: {loss_smooth.item()} vs {loss_no_smooth.item()}"
        )
        print(f"  [PASS] label_smoothing: no_smooth={loss_no_smooth.item():.4f}, smooth={loss_smooth.item():.4f}")

    def test_batch_processing(self):
        """Test with batched inputs."""
        batch_size = 4
        policy_chosen = torch.tensor([-1.0, -2.0, -1.5, -3.0])
        policy_rejected = torch.tensor([-3.0, -1.0, -4.0, -2.0])
        ref_chosen = torch.tensor([-2.0, -2.0, -2.0, -2.0])
        ref_rejected = torch.tensor([-2.0, -2.0, -2.0, -2.0])

        loss, c_rew, r_rew = dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
        )

        assert loss.dim() == 0, "Loss should be scalar"
        assert c_rew.dim() == 0, "Rewards should be scalar"
        print(f"  [PASS] batch_processing: loss={loss.item():.4f}")

    def test_gradient_flow(self):
        """Verify gradients flow through the loss."""
        policy_chosen = torch.tensor([-1.0], requires_grad=True)
        policy_rejected = torch.tensor([-3.0], requires_grad=True)
        ref_chosen = torch.tensor([-2.0])
        ref_rejected = torch.tensor([-2.0])

        loss, _, _ = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
        loss.backward()

        assert policy_chosen.grad is not None, "Gradient should flow to policy_chosen"
        assert policy_rejected.grad is not None, "Gradient should flow to policy_rejected"
        # Gradient for chosen should be negative (increasing chosen logps decreases loss)
        assert policy_chosen.grad.item() < 0, f"Chosen grad should be negative, got {policy_chosen.grad.item()}"
        # Gradient for rejected should be positive (increasing rejected logps increases loss)
        assert policy_rejected.grad.item() > 0, f"Rejected grad should be positive, got {policy_rejected.grad.item()}"
        print(f"  [PASS] gradient_flow: chosen_grad={policy_chosen.grad.item():.6f}, rejected_grad={policy_rejected.grad.item():.6f}")


def run_all_tests():
    print("=" * 60)
    print("Running DPO Loss Unit Tests")
    print("=" * 60)

    print("\n--- TestComputeLogProbs ---")
    t1 = TestComputeLogProbs()
    t1.test_basic_correctness()
    t1.test_manual_computation()
    t1.test_zero_mask()

    print("\n--- TestDPOLoss ---")
    t2 = TestDPOLoss()
    t2.test_chosen_preferred_loss_is_low()
    t2.test_rejected_preferred_loss_is_high()
    t2.test_reference_free()
    t2.test_ipo_loss()
    t2.test_reward_margin()
    t2.test_label_smoothing()
    t2.test_batch_processing()
    t2.test_gradient_flow()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
