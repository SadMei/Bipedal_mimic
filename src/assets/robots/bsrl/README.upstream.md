# BSRL robot model bundle

This repository is the canonical, self-contained BSRL robot asset bundle:

- `urdf/`: serial robot URDF and a static closed-ankle preview.
- `xml/`: serial and explicit closed-chain MuJoCo models.
- `meshes/`: all geometry referenced by the URDF and XML files.

The model files use relative mesh paths. Keep these three directories together
when copying the bundle; no runtime path points to another repository.

## Install into Bipedal_AMP

Bipedal training intentionally keeps its existing model location. From this
repository, copy the serial bundle into:

```text
Bipedal_AMP/source/legged_lab/legged_lab/data/Robots/BSRL_urdf/
```

The optional explicit closed-ankle tools use the same `xml/` and `meshes/`
contents under:

```text
Bipedal_AMP/source/legged_lab/legged_lab/data/Robots/BSRL_closed_ankle/
```

The Bipedal repository deliberately does not track mesh payloads. Copying the
asset directories locally supplies them without changing its Git policy.

## Install into GMR

Copy `urdf/`, `xml/`, and `meshes/` into:

```text
GMR/assets/BSRL_urdf/
```

GMR uses `assets/BSRL_urdf/xml/bsrl_scene.xml` for the BSRL viewer and does not
reference this repository at runtime.
