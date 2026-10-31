"""Exercise the adapter boundary without importing torch or loading weights."""

from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from utils.miku_multiview_utils import MikuMultiviewDiffusionNet, prepare_miku_reference, load_miku_adapter_weights


class FakeGenerator:
    def __init__(self, device):
        self.device = device

    def manual_seed(self, seed):
        self.seed = seed
        return self


class FakeModel:
    def __init__(self):
        self.lora_layer = FakeLoRALayer()
        self.saved_layer = FakeSavedLayer()
        self.transformer_blocks = []
        self.single_transformer_blocks = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        instance = cls()
        instance.path, instance.kwargs = path, kwargs
        return instance

    def eval(self):
        self.evaluating = True

    def state_dict(self):
        return {"x_embedder.weight": FakeTensor(np.zeros((4, 8), dtype=np.float32))}

    def named_modules(self):
        return [("block", self.lora_layer), ("x_embedder", self.saved_layer)]

    def requires_grad_(self, enabled):
        self.grad_enabled = enabled


class FakeLoRALayer:
    def __init__(self):
        self.scaling, self.r, self.lora_alpha = {"texture": 1.0}, {"texture": 2}, {"texture": 4}
        self.use_rslora = {"texture": False}
        self.active_adapters, self.disable_adapters = [], True

    def enable_adapters(self, enabled):
        self.disable_adapters = not enabled


class FakeSavedLayer:
    def __init__(self):
        self.modules_to_save = {"texture": object()}
        self.active_adapters, self.disable_adapters = [], True

    def enable_adapters(self, enabled):
        self.disable_adapters = not enabled

    def set_adapter(self, name):
        self.active_adapters = [name]


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.shape, self.dtype = self.values.shape, self.values.dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, **kwargs):
        return self


class FakeFluxPipeline:
    vae_scale_factor = 8

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        instance = cls()
        instance.path, instance.kwargs = path, kwargs
        instance.transformer = kwargs["transformer"]
        instance.vae = FakeModel()
        return instance

    def lora_state_dict(self, path, **kwargs):
        self.lora_path, self.lora_kwargs = path, kwargs
        state = {
            "transformer.block.lora_A.weight": FakeTensor(np.ones((2, 4))),
            "transformer.block.lora_B.weight": FakeTensor(np.full((4, 2), 2)),
            "transformer.x_embedder.weight": FakeTensor(np.full((4, 8), 3)),
        }
        return state, {"transformer.modules_to_save": ["x_embedder"]}

    def load_lora_into_transformer(self, state, transformer, adapter_name, **kwargs):
        self.lora_kwargs["adapter_name"] = adapter_name
        transformer.adapter_state = {key.removeprefix("transformer."): value for key, value in state.items()}

    def to(self, device):
        self.device = device
        return self

    def enable_model_cpu_offload(self, **kwargs):
        self.offload_kwargs = kwargs

    def __call__(self, images, **kwargs):
        self.input_images, self.call_kwargs = images, kwargs
        return SimpleNamespace(images=[images[1].copy()])


def fake_torch_module():
    module = ModuleType("torch")
    for dtype in ("float16", "bfloat16", "float32"):
        setattr(module, dtype, dtype)
    module.Generator = FakeGenerator
    module.inference_mode = nullcontext
    module.equal = lambda first, second: np.array_equal(first.values, second.values)
    module.device = lambda value: SimpleNamespace(type=value.split(":")[0],
                                                  index=int(value.split(":")[1]) if ":" in value else None)
    return module


class AdapterTests(unittest.TestCase):
    def setUp(self):
        pipeline_module = ModuleType("miku.pipeline_drop")
        pipeline_module.FluxMVKleinPipeline = FakeFluxPipeline
        transformer_module = ModuleType("miku.drop_transformer")
        transformer_module.Flux2DropConditionModel = FakeModel
        peft_module = ModuleType("peft.utils.save_and_load")
        peft_module.get_peft_model_state_dict = lambda model, **kwargs: model.adapter_state
        self.peft_module = peft_module
        diffusers_peft_module = ModuleType("diffusers.utils.peft_utils")

        def set_weights(model, names, weights):
            model.adapter_setting = (names, weights)
            model.lora_layer.active_adapters = names
            model.lora_layer.scaling["texture"] = weights[0] * 4 / 2

        diffusers_peft_module.set_weights_and_activate_adapters = set_weights
        peft_tuners = ModuleType("peft.tuners.tuners_utils")
        peft_tuners.BaseTunerLayer = FakeLoRALayer
        peft_other = ModuleType("peft.utils.other")
        peft_other.ModulesToSaveWrapper = FakeSavedLayer
        self.patcher = patch.dict("sys.modules", {
            "torch": fake_torch_module(), "miku.pipeline_drop": pipeline_module,
            "miku.drop_transformer": transformer_module, "peft.utils.save_and_load": peft_module,
            "diffusers.utils.peft_utils": diffusers_peft_module, "peft.tuners.tuners_utils": peft_tuners,
            "peft.utils.other": peft_other})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.config = SimpleNamespace(
            device="cuda:1", resolution=96, miku_pretrained_path="/weights/klein",
            miku_lora_path="/weights/mv.safetensors", miku_lora_scale=0.75,
            miku_steps=13, miku_guidance_scale=2.25, miku_seed=47,
            miku_offload="model", miku_local_files_only=True,
            miku_prompt="configured prompt", miku_max_canvas_size=2048)

    def make_conditions(self):
        normals, positions, masks = [], [], []
        for index in range(6):
            normal = Image.new("RGB", (96, 96), (20 + index * 20, 60, 100))
            position = Image.new("RGB", (96, 96), (40 + index * 20, 80, 120))
            mask = Image.new("L", (96, 96))
            mask.paste(255, (16, 16, 80, 80))
            normals.append(normal)
            positions.append(position)
            masks.append(mask)
        return normals + positions, masks

    def test_complete_fake_call_and_debug_artifacts(self):
        conditions, masks = self.make_conditions()
        with tempfile.TemporaryDirectory() as folder:
            self.config.miku_debug_dir = folder
            adapter = MikuMultiviewDiffusionNet(self.config)
            result = adapter([Image.new("RGB", (160, 80), (0, 70, 130))], conditions,
                             prompt="explicit prompt", masks=masks)
            pipeline = adapter.pipeline
            self.assertEqual(pipeline.path, self.config.miku_pretrained_path)
            self.assertTrue(pipeline.kwargs["local_files_only"])
            self.assertEqual(pipeline.lora_path, "/weights")
            self.assertEqual(pipeline.lora_kwargs["weight_name"], "mv.safetensors")
            self.assertTrue(pipeline.lora_kwargs["local_files_only"])
            self.assertEqual(pipeline.lora_kwargs["adapter_name"], "texture")
            self.assertEqual(pipeline.transformer.adapter_setting, (["texture"], [0.75]))
            self.assertEqual(pipeline.transformer.lora_layer.scaling["texture"], 1.5)
            self.assertEqual(pipeline.transformer.saved_layer.active_adapters, ["texture"])
            self.assertFalse(pipeline.transformer.saved_layer.disable_adapters)
            self.assertFalse(pipeline.transformer.grad_enabled)
            self.assertEqual(pipeline.offload_kwargs, {"gpu_id": 1, "device": "cuda"})
            self.assertTrue(pipeline.transformer.evaluating)
            self.assertEqual(adapter.adapter_report, {
                "tensor_count": 3, "full_parameter_keys": ["x_embedder.weight"], "verified": True})
            self.assertEqual(len(pipeline.input_images), 2)
            self.assertEqual(pipeline.input_images[0].size, (1024, 1024))
            control = pipeline.input_images[1]
            self.assertEqual(control.size, tuple(adapter.last_metadata["model_canvas_size"]))
            self.assertEqual((pipeline.call_kwargs["width"], pipeline.call_kwargs["height"]), control.size)
            self.assertEqual(pipeline.call_kwargs["num_inference_steps"], 13)
            self.assertEqual(pipeline.call_kwargs["guidance_scale"], 2.25)
            self.assertEqual(pipeline.call_kwargs["prompt"], "explicit prompt")
            generator = pipeline.call_kwargs["generator"]
            self.assertEqual((generator.device, generator.seed), ("cuda:1", 47))
            self.assertEqual(set(result), {"albedo"})
            self.assertEqual(len(result["albedo"]), 6)
            for index, view in enumerate(result["albedo"]):
                self.assertEqual(view.size, (96, 96))
                self.assertEqual(view.getpixel((48, 48)), (30 + index * 20, 70, 110))
            expected = {"control.png", "reference.png", "generated.png", "packing.json"}
            expected |= {f"view_{index:02d}.png" for index in range(6)}
            self.assertEqual({path.name for path in Path(folder).iterdir()}, expected)

    def test_config_prompt_and_batch_constraints(self):
        self.config.miku_offload = "none"
        adapter = MikuMultiviewDiffusionNet(self.config)
        conditions, masks = self.make_conditions()
        reference = Image.new("RGB", (96, 96), "white")
        adapter(reference, conditions, masks=masks)
        self.assertEqual(adapter.pipeline.call_kwargs["prompt"], "configured prompt")
        self.assertEqual(adapter.pipeline.device, "cuda:1")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            adapter([reference, reference], conditions, masks=masks)
        with self.assertRaisesRegex(ValueError, "one string prompt"):
            adapter(reference, conditions, masks=masks, prompt=["one", "two"])
        with patch.object(FakeFluxPipeline, "vae_scale_factor", 16):
            with self.assertRaisesRegex(ValueError, "16-pixel latent token grid"):
                MikuMultiviewDiffusionNet(self.config)

    def test_offline_lora_directory_always_has_explicit_weight_name(self):
        with tempfile.TemporaryDirectory() as folder:
            self.config.miku_lora_path = folder
            adapter = MikuMultiviewDiffusionNet(self.config)
            self.assertEqual(adapter.pipeline.lora_path, folder)
            self.assertEqual(adapter.pipeline.lora_kwargs["weight_name"], "pytorch_lora_weights.safetensors")
            self.config.miku_lora_weight_name = "student_lora_diffusers.safetensors"
            adapter = MikuMultiviewDiffusionNet(self.config)
            self.assertEqual(adapter.pipeline.lora_kwargs["weight_name"], "student_lora_diffusers.safetensors")

    def test_full_weights_are_never_silently_discarded(self):
        pipeline = FakeFluxPipeline.from_pretrained("/weights", transformer=FakeModel())
        original = pipeline.lora_state_dict

        def bad_metadata(*args, **kwargs):
            state, metadata = original(*args, **kwargs)
            return state, {}

        with patch.object(pipeline, "lora_state_dict", side_effect=bad_metadata):
            with self.assertRaisesRegex(ValueError, "not declared"):
                load_miku_adapter_weights(pipeline, "/weights", "weights.safetensors")
        with patch.object(self.peft_module, "get_peft_model_state_dict", return_value={}):
            with self.assertRaisesRegex(ValueError, "parameter mismatch"):
                load_miku_adapter_weights(pipeline, "/weights", "weights.safetensors")

        def corrupt_export(model, **kwargs):
            state = dict(model.adapter_state)
            state["x_embedder.weight"] = FakeTensor(np.zeros((4, 8)))
            return state

        with patch.object(self.peft_module, "get_peft_model_state_dict", side_effect=corrupt_export):
            with self.assertRaisesRegex(ValueError, "not restored exactly"):
                load_miku_adapter_weights(pipeline, "/weights", "weights.safetensors")

    def test_reference_fit_preserves_aspect_and_existing_square(self):
        reference = prepare_miku_reference(Image.new("RGB", (160, 80), (0, 70, 130)))
        pixels = np.asarray(reference)
        self.assertTrue(np.all(pixels[:256] == 255))
        self.assertTrue(np.all(pixels[256:768] == [0, 70, 130]))
        self.assertTrue(np.all(pixels[768:] == 255))
        y, x = np.mgrid[:1024, :1024]
        original = Image.fromarray(np.stack([x % 255, y % 255, (x + y) % 255], axis=-1).astype(np.uint8))
        np.testing.assert_array_equal(prepare_miku_reference(original), original)
        stretched = prepare_miku_reference(Image.new("RGB", (160, 80), (0, 70, 130)), mode="stretch")
        self.assertTrue(np.all(np.asarray(stretched) == [0, 70, 130]))

    def test_reference_alpha_is_cropped_and_composited_once(self):
        rgba = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
        rgba.paste((0, 80, 160, 128), (16, 32, 80, 64))
        reference = prepare_miku_reference(rgba)
        self.assertEqual(reference.getpixel((512, 512)), (127, 167, 207))
        self.assertEqual(reference.getpixel((512, 0)), (255, 255, 255))
        with self.assertRaisesRegex(ValueError, "no visible pixels"):
            prepare_miku_reference(Image.new("RGBA", (16, 16)))


if __name__ == "__main__":
    unittest.main()
