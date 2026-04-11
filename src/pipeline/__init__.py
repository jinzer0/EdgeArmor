def __getattr__(name):
    if name == "DeepfakeDetectionPipeline":
        from .deepfake_pipeline import DeepfakeDetectionPipeline

        return DeepfakeDetectionPipeline

    if name == "select_faces":
        from .face_selector import select_faces

        return select_faces

    if name == "ForensicsAdapterInfer":
        from .forensics_adapter_infer import ForensicsAdapterInfer

        return ForensicsAdapterInfer

    if name == "build_face_data_dict":
        from .preprocess import build_face_data_dict

        return build_face_data_dict

    if name == "build_face_image_tensor":
        from .preprocess import build_face_image_tensor

        return build_face_image_tensor

    if name == "create_if_boundary":
        from .preprocess import create_if_boundary

        return create_if_boundary

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
