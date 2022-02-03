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
import {ArcRotateCamera, Nullable, Quaternion, Scene} from "@babylonjs/core";
import {Color3, Vector3} from "@babylonjs/core/Maths";
import {Engine} from "@babylonjs/core/Engines";
import {Camera} from "@babylonjs/core";
import {DebugInfo} from "./helper/debug";

// Debug
import "@babylonjs/core/Debug";
import "@babylonjs/gui";
import "@babylonjs/inspector";

import * as Comlink from "comlink";
import {Poses} from "./worker/pose-processing";
import {Clock} from "./helper/clock";
import {VRMManager} from "v3d-core/dist/src/importer/babylon-vrm-loader/src";
import {HAND_LANDMARKS_BONE_MAPPING} from "./helper/landmark";
import {HumanoidBone} from "v3d-core/dist/src/importer/babylon-vrm-loader/src/humanoid-bone";
import {KeysMatching, LR} from "./helper/utils";
import {
    CloneableQuaternionMap,
    cloneableQuaternionToQuaternion,
} from "./helper/quaternion";
import {Holistic} from "@mediapipe/holistic";
import {BoneOptions, BoneState, HolisticState} from "./v3d-web";

const IS_DEBUG = false;
const clock = new Clock(), textDecode = new TextDecoder();
export let debugInfo: Nullable<DebugInfo>;

// Can only have one VRM model at this time
export async function createScene(
    engine: Engine,
    workerPose: Nullable<Comlink.Remote<Poses>>,
    boneState: BoneState,
    boneOptions: BoneOptions,
    holistic: Holistic,
    holisticState: HolisticState,
    vrmFile: File | string,
    videoElement: HTMLVideoElement): Promise<Nullable<[V3DCore, VRMManager]>> {

    if (!workerPose) return null;

    // Create v3d core
    const v3DCore = new V3DCore(engine, new Scene(engine));
    await v3DCore.AppendAsync('', vrmFile);

    // Get managers
    const vrmManager = v3DCore.getVRMManagerByURI((vrmFile as File).name ? (vrmFile as File).name : (vrmFile as string));

    // Camera
    // v3DCore.attachCameraTo(vrmManager);
    const mainCamera = (v3DCore.mainCamera as ArcRotateCamera);
    mainCamera.setPosition(new Vector3(0, 1.05, 4.5));
    mainCamera.setTarget(
        vrmManager.rootMesh.getWorldMatrix().getTranslation().subtractFromFloats(0, -1.25, 0));
    mainCamera.fovMode = Camera.FOVMODE_HORIZONTAL_FIXED;

    // Lights and Skybox
    v3DCore.addAmbientLight(new Color3(1, 1, 1));
    v3DCore.setBackgroundColor(Color3.FromHexString('#e7a2ff'));

    v3DCore.renderingPipeline.depthOfFieldEnabled = false;

    // Pose web worker
    await workerPose.setBonesHierarchyTree(vrmManager.transformNodeTree);

    // Disable auto animation
    v3DCore.springBonesAutoUpdate = false;

    // Update functions
    v3DCore.updateBeforeRenderFunction(
        () => {
            // Half input fps. This version of Holistic is heavy on CPU time.
            // Wait until they fix web worker (https://github.com/google/mediapipe/issues/2506).
            if (holisticState.holisticUpdate && holisticState.ready && !videoElement.paused && videoElement.readyState > 2) {
                holistic.send({image: videoElement})
            }
            holisticState.holisticUpdate = !holisticState.holisticUpdate;
        }
    );
    v3DCore.updateAfterRenderFunction(
        () => {
            if (boneState.bonesNeedUpdate) {
                updatePose(vrmManager, boneState, boneOptions);
                updateSpringBones(vrmManager);
                boneState.bonesNeedUpdate = false;
            }
        }
    );

    // Render loop
    engine.runRenderLoop(() => {
        v3DCore.scene?.render();
    });

    // Model Transformation
    vrmManager.rootMesh.rotationQuaternion = Quaternion.RotationYawPitchRoll(0, 0, 0);

    // Debug
    if (IS_DEBUG && v3DCore.scene) {
        debugInfo = new DebugInfo(v3DCore.scene);
    }

    engine.hideLoadingUI();

    return [v3DCore, vrmManager];
}

export function updateSpringBones(vrmManager: VRMManager) {
    const deltaTime = clock.getDelta() * 1000;
    vrmManager.update(deltaTime);
}

export function updateBuffer(data: Uint8Array, boneState: BoneState) {
    let jsonStr = textDecode.decode(data);
    let boneRotationsData: CloneableQuaternionMap = JSON.parse(jsonStr);
    boneState.boneRotations = boneRotationsData;
    boneState.bonesNeedUpdate = true;
    (data as any) = null;
    (jsonStr as any) = null;
}

export function updatePose(
    vrmManager: VRMManager,
    boneState: BoneState,
    boneOptions: BoneOptions
) {
    // Wait for buffer to fill
    if (!boneState.boneRotations) return;

    const resultBoneRotations = boneState.boneRotations;

    vrmManager.morphing('A', resultBoneRotations['mouth'].x);

    const resetExpressions = () => {
        vrmManager.morphing('Neutral', 0);
        vrmManager.morphing('Happy', 0);
        vrmManager.morphing('Joy', 0);
        vrmManager.morphing('Angry', 0);
        vrmManager.morphing('Sad', 0);
        vrmManager.morphing('Sorrow', 0);
        vrmManager.morphing('Relaxed', 0);
        vrmManager.morphing('Fun', 0);
        vrmManager.morphing('Surprised', 0);
    }

    // Update expression
    resetExpressions();
    switch (boneOptions.expression) {
        case "Angry":
            vrmManager.morphing('Angry', 1);
            break;
        case "Happy":
            vrmManager.morphing('Happy', 1);
            vrmManager.morphing('Joy', 1);
            break;
        case "Relaxed":
            vrmManager.morphing('Relaxed', 1);
            vrmManager.morphing('Fun', 1);
            break;
        case "Sad":
            vrmManager.morphing('Sad', 1);
            vrmManager.morphing('Sorrow', 1);
            break;
        case "Surprised":
            vrmManager.morphing('Surprised', 1);
            break;
        case "Neutral": // fall through
        default:
            vrmManager.morphing('Neutral', 1);
            break;
    }

    if (boneOptions.blinkLinkLR) {
        vrmManager.morphing('Blink', resultBoneRotations['blink'].z)
    } else {
        vrmManager.morphing('Blink_L', resultBoneRotations['blink'].x)
        vrmManager.morphing('Blink_R', resultBoneRotations['blink'].y)
    }

    if (vrmManager.humanoidBone.leftEye)
        vrmManager.humanoidBone.leftEye.rotationQuaternion = cloneableQuaternionToQuaternion(
            resultBoneRotations['iris']);
    if (vrmManager.humanoidBone.rightEye)
        vrmManager.humanoidBone.rightEye.rotationQuaternion = cloneableQuaternionToQuaternion(
            resultBoneRotations['iris']);

    for (const d of LR) {
        for (const k of Object.keys(HAND_LANDMARKS_BONE_MAPPING)) {
            const key = d + k as keyof Omit<HumanoidBone, KeysMatching<HumanoidBone, Function>>;
            if (vrmManager.humanoidBone[key]) {
                vrmManager.humanoidBone[key]!.rotationQuaternion = cloneableQuaternionToQuaternion(
                    resultBoneRotations[key]);
            }
        }
    }

    vrmManager.humanoidBone.hips.rotationQuaternion = cloneableQuaternionToQuaternion(
        resultBoneRotations['hips']);
    vrmManager.humanoidBone.spine.rotationQuaternion = cloneableQuaternionToQuaternion(
        resultBoneRotations['spine']);
    vrmManager.humanoidBone.neck.rotationQuaternion = cloneableQuaternionToQuaternion(
        resultBoneRotations['neck']);
    vrmManager.humanoidBone.head.rotationQuaternion = cloneableQuaternionToQuaternion(
        resultBoneRotations['head']);

    // Arms
    for (const k of LR) {
        const upperArmKey = `${k}UpperArm` as unknown as keyof Omit<HumanoidBone, KeysMatching<HumanoidBone, Function>>;
        const lowerArmKey = `${k}LowerArm` as unknown as keyof Omit<HumanoidBone, KeysMatching<HumanoidBone, Function>>;
        vrmManager.humanoidBone[upperArmKey]!.rotationQuaternion = cloneableQuaternionToQuaternion(
            resultBoneRotations[upperArmKey]);
        vrmManager.humanoidBone[lowerArmKey]!.rotationQuaternion = cloneableQuaternionToQuaternion(
            resultBoneRotations[lowerArmKey]);
        // Legs
        const upperLegKey = `${k}UpperLeg` as unknown as keyof Omit<HumanoidBone, KeysMatching<HumanoidBone, Function>>;
        const lowerLegKey = `${k}LowerLeg` as unknown as keyof Omit<HumanoidBone, KeysMatching<HumanoidBone, Function>>;
        vrmManager.humanoidBone[upperLegKey]!.rotationQuaternion = cloneableQuaternionToQuaternion(
            resultBoneRotations[upperLegKey]);
        vrmManager.humanoidBone[lowerLegKey]!.rotationQuaternion = cloneableQuaternionToQuaternion(
            resultBoneRotations[lowerLegKey]);
        // Feet
        const footKey = `${k}Foot` as unknown as keyof Omit<HumanoidBone, KeysMatching<HumanoidBone, Function>>;
        vrmManager.humanoidBone[footKey]!.rotationQuaternion = cloneableQuaternionToQuaternion(
            resultBoneRotations[footKey]);
    }
}
