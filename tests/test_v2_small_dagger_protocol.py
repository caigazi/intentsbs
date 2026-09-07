import json
from pathlib import Path
import unittest


def parse_spec(value):
    n_agents, seed = value.split(":")
    return int(n_agents), int(seed)


class SmallDaggerProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / "SMALL_DAGGER_PROTOCOL.json"
        cls.protocol = json.loads(path.read_text(encoding="utf-8"))

    def test_validation_never_enters_training(self):
        self.assertFalse(self.protocol["fixed_validation"]["enters_training"])
        validation = {
            parse_spec(value) for value in
            self.protocol["fixed_validation"]["scene_specs"]
        }
        training = {
            parse_spec(value)
            for values in self.protocol["training_rollouts"][
                "round_scene_specs"].values()
            for value in values
        }
        self.assertTrue(validation.isdisjoint(training))

    def test_every_round_is_n8_anchored_and_multi_n(self):
        for values in self.protocol["training_rollouts"][
                "round_scene_specs"].values():
            counts = [parse_spec(value)[0] for value in values]
            self.assertEqual(len(counts), 16)
            self.assertEqual(counts.count(8), 10)
            for n_agents in range(2, 8):
                self.assertIn(n_agents, counts)

    def test_round_and_window_limits_are_frozen(self):
        self.assertEqual(self.protocol["maximum_rounds"], 3)
        self.assertEqual(self.protocol["rollout_horizon"], 32)
        self.assertEqual(
            self.protocol["maximum_selected_states_per_scene"], 4)


if __name__ == "__main__":
    unittest.main()
