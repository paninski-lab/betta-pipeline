def lp_train(video_folder, output_folder, cfg_file, ckpt_file):
    import torch
    import omegaconf
    from omegaconf import dictconfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode
    from omegaconf.listconfig import ListConfig
    import collections, typing

    torch.serialization.add_safe_globals([
        collections.defaultdict,
        typing.Any,
        dict,
        dictconfig.DictConfig,
        ContainerMetadata,
        AnyNode,
        Metadata,
        ListConfig,
        list,
        int
    ])

    from pathlib import Path
    from lightning_pose.utils.predictions import predict_single_video

    video_folder = Path(video_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(video_folder.glob("*.mp4"))

    if not video_paths:
        raise RuntimeError(f"No .mp4 files found in {video_folder}")

    for vid_path in video_paths:
        base = vid_path.stem
        output_file = output_folder / f"{base}.csv"

        print(f"Processing {vid_path.name} → {output_file}")

        predict_single_video(
            cfg_file=cfg_file,
            video_file=str(vid_path),
            preds_file=str(output_file),
            ckpt_file=ckpt_file,
        )

    print(f"\n✅ All videos processed and predictions saved to: {output_folder}")