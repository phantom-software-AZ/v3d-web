/*
Copyright (C) 2021  The v3d Authors.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, version 3.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

import {V3DCore} from "v3d-core/dist/src";
import {ArcRotateCamera, Mesh, Nullable, Quaternion, Scene} from "@babylonjs/core";
import {Color3, Vector3} from "@babylonjs/core/Maths";
import {Engine} from "@babylonjs/core/Engines";
import {DebugInfo} from "./helper/debug";

// Debug
import "@babylonjs/core/Debug";
import "@babylonjs/gui";
import "@babylonjs/inspector";
const IS_DEBUG = true;
export let debugInfo: Nullable<DebugInfo>;

// Can only have one VRM model at this time
export async function createScene(engine: Engine) {
    const vrmFile = 'testfiles/2078913627571329107.vrm';

    // Create v3d core
    const v3DCore = new V3DCore(engine, new Scene(engine));
    v3DCore.transparentBackground();
    await v3DCore.AppendAsync('', vrmFile);

    // Get managers
    const vrmManager = v3DCore.getVRMManagerByURI(vrmFile);

    // Camera
    v3DCore.attachCameraTo(vrmManager);
    (v3DCore.mainCamera as ArcRotateCamera).setPosition(new Vector3(0, 0, -5));
    (v3DCore.mainCamera as ArcRotateCamera).setTarget(Vector3.Zero());

    // Lights
    v3DCore.addAmbientLight(new Color3(1, 1, 1));

    // Lock camera target
    v3DCore.scene?.onBeforeRenderObservable.add(() => {
        vrmManager.cameras[0].setTarget(vrmManager.rootMesh.getAbsolutePosition());
    });

    // Render loop
    engine.runRenderLoop(() => {
        v3DCore.scene?.render();
    });

    // Model Transformation
    vrmManager.rootMesh.translate(new Vector3(1, 0, 0), 1);
    vrmManager.rootMesh.rotation = new Vector3(0, 135, 0);

    // Work with HumanoidBone
    vrmManager.humanoidBone.leftUpperArm.addRotation(0, -0.5, 0);
    vrmManager.humanoidBone.head.addRotation(0.1, 0, 0);

    // Work with BlendShape(MorphTarget)
    // vrmManager.morphing('Joy', 1.0);
    // TODO: Debug only
    // @ts-ignore
    window['vrmManager'] = vrmManager;
    // @ts-ignore
    vrmManager.r = Quaternion.RotationYawPitchRoll;

    // Debug
    if (IS_DEBUG && v3DCore.scene) debugInfo = new DebugInfo(v3DCore.scene);

    return vrmManager;
}
