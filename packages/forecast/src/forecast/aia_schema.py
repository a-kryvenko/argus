"""AIA193 sector feature schema shared by the deployed residual model."""
def feature_columns(features):
    control = ["dlinear_v", "lead_hours", "calendar_sin", "calendar_cos",
               "aia_age_hours", "aia_valid_fraction", "aia_b0_deg"]
    current = [c for c in features if c.startswith("aia_area_")]
    changes = [c for c in features if c.startswith(("aia_delta_", "aia_overlap_"))
               or c.endswith("_separation_h")]
    if not current or not changes:
        raise ValueError("Missing daily AIA area/change features")
    return control + current + changes
