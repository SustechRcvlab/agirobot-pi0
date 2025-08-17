import logging
import jax
import etils.epath as epath
import openpi.training.sharding as sharding
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils
import openpi.training.config as _config

if __name__ == "__main__":
    config = _config.get_config("pi0_base_agibot_lora")
    print(config)

    jax.config.update(
        "jax_compilation_cache_dir",
        str(epath.Path("~/.cache/jax").expanduser())
    )

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        # num_workers=config.num_workers,
        num_workers=0,
        shuffle=True,
    )

    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
