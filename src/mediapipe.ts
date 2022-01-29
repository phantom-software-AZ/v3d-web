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

import * as Comlink from "comlink";
import {
    ControlPanel,
    FPS,
    OptionMap,
    Slider,
    StaticText,
    Toggle
} from "@mediapipe/control_utils";
import {
    FACEMESH_FACE_OVAL,
    FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW,
    FACEMESH_LEFT_IRIS, FACEMESH_LIPS,
    FACEMESH_RIGHT_EYE, FACEMESH_RIGHT_EYEBROW,
    FACEMESH_RIGHT_IRIS,
    HAND_CONNECTIONS,
    Holistic,
    NormalizedLandmark,
    NormalizedLandmarkList,
    Options, POSE_CONNECTIONS,
    POSE_LANDMARKS, POSE_LANDMARKS_LEFT, POSE_LANDMARKS_RIGHT,
    Results
} from "@mediapipe/holistic";
import {Data, drawConnectors, drawLandmarks, lerp} from "@mediapipe/drawing_utils";
import {Poses} from "./worker/pose-processing";
import {debugInfo, updateBuffer} from "./core";
import {Vector3, Nullable} from "@babylonjs/core";
import {VRMManager} from "v3d-core/dist/src/importer/babylon-vrm-loader/src";

function removeElements(
    landmarks: NormalizedLandmarkList, elements: number[]) {
    for (const element of elements) {
        delete landmarks[element];
    }
}

function removeLandmarks(results: Results) {
    if (results.poseLandmarks) {
        removeElements(
            results.poseLandmarks,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22]);
    }
}

function connect(
    ctx: CanvasRenderingContext2D,
    connectors:
        Array<[NormalizedLandmark, NormalizedLandmark]>):
    void {
    const canvas = ctx.canvas;
    for (const connector of connectors) {
        const from = connector[0];
        const to = connector[1];
        if (from && to) {
            if (from.visibility && to.visibility &&
                (from.visibility < 0.1 || to.visibility < 0.1)) {
                continue;
            }
            ctx.beginPath();
            ctx.moveTo(from.x * canvas.width, from.y * canvas.height);
            ctx.lineTo(to.x * canvas.width, to.y * canvas.height);
            ctx.stroke();
        }
    }
}

const holisticOptions = {
    selfieMode: true,
    modelComplexity: 1,
    useCpuInference: false,
    cameraOn: true,
    smoothLandmarks: true,
    enableSegmentation: false,
    smoothSegmentation: false,
    refineFaceLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.55,
    effect: 'background',
};

export function onResults(
    results: Results,
    vrmManager: VRMManager,
    workerPose: Comlink.Remote<Poses>,
    activeEffect: string,
    fpsControl: Nullable<FPS>
): void {
    // notify loaded.
    document.body.classList.add('loaded');

    // TODO: !!!toggle on before release!!!
    // @ts-ignore: delete camera input to prevent accidental paint
    delete results.image;

    if (!holisticOptions.cameraOn) return;

    // Worker process
    workerPose.process(
        (({segmentationMask, image, ...o}) => o)(results),    // Remove canvas properties
        Comlink.proxy(updateBuffer)
    )
        .then(async (r) => {
            // if (debugInfo) {
            //     const resultPoseLandmarks = await workerPose.cloneablePoseLandmarks;
            //     const resultFaceNormal = await workerPose.faceNormal;
            //     const resultIrisQuaternions = await workerPose.irisQuaternion;
            //     const resultFaceMeshIndexLandmarks = await workerPose.faceMeshLandmarkIndexList;
            //     const resultFaceMeshLandmarks = await workerPose.faceMeshLandmarkList;
            //     const resultLeftHandLandmarks = await workerPose.cloneableLeftHandLandmarks;
            //     const resultRightHandLandmarks = await workerPose.cloneableRightHandLandmarks;
            //     const resultLeftHandNormals = await workerPose.leftHandNormals;
            //     const resultRightHandNormals = await workerPose.rightHandNormals;
            //     const resultPoseNormals = await workerPose.poseNormals;
            //     debugInfo.updatePoseLandmarkSpheres(resultPoseLandmarks);
            //     debugInfo.updateFaceNormalArrows(
            //         [resultFaceNormal], resultPoseLandmarks);
            //     debugInfo.updateFaceMeshLandmarkSpheres(
            //         resultFaceMeshIndexLandmarks, resultFaceMeshLandmarks);
            //     // debugInfo.updateHandLandmarkSpheres(resultLeftHandLandmarks, true);
            //     // debugInfo.updateHandLandmarkSpheres(resultRightHandLandmarks, false);
            //     debugInfo.updateIrisQuaternionArrows(
            //         resultIrisQuaternions, resultPoseLandmarks, resultFaceNormal);
                // debugInfo.updateHandWristNormalArrows(
                //     resultLeftHandBoneRotations, resultRightHandBoneRotations, resultPoseLandmarks);
                // debugInfo.updateHandNormalArrows(
                //     resultLeftHandNormals, resultRightHandNormals, resultPoseLandmarks);
            //     debugInfo.updatePoseNormalArrows(resultPoseNormals, resultPoseLandmarks);
            // }

            workerPose.midHipPos.then((v) => {
                if (v)
                    vrmManager.rootMesh.position = new Vector3(-v.x, v.y, -v.z);
            });
        });

    // Remove landmarks we don't want to draw.
    removeLandmarks(results);

    // Update the frame rate.
    if (fpsControl) fpsControl.tick();

    // Get canvas context
    const videoCanvasElement =
        document.getElementById('video-canvas') as HTMLCanvasElement;
    const videoCanvasCtx = videoCanvasElement.getContext('2d');
    if (!videoCanvasCtx) return;

    // Draw the overlays.
    videoCanvasCtx.save();
    videoCanvasCtx.clearRect(0, 0, videoCanvasElement.width, videoCanvasElement.height);

    // Draw safe area
    videoCanvasCtx.fillStyle = 'rgba(34,34,34,0.5)';
    videoCanvasCtx.strokeStyle = 'red';
    videoCanvasCtx.lineWidth = 3;
    const paddingX = 0.04, paddingY = 0.08;
    videoCanvasCtx.fillRect(videoCanvasElement.width * paddingX,
        videoCanvasElement.height * paddingY,
        videoCanvasElement.width * (1 - 2 * paddingX),
        videoCanvasElement.height * (1 - 2 * paddingY));
    videoCanvasCtx.strokeRect(videoCanvasElement.width * paddingX,
        videoCanvasElement.height * paddingY,
        videoCanvasElement.width * (1 - 2 * paddingX),
        videoCanvasElement.height * (1 - 2 * paddingY));

    if (results.segmentationMask) {
        videoCanvasCtx.drawImage(
            results.segmentationMask, 0, 0, videoCanvasElement.width,
            videoCanvasElement.height);

        // Only overwrite existing pixels.
        if (activeEffect === 'mask' || activeEffect === 'both') {
            videoCanvasCtx.globalCompositeOperation = 'source-in';
            // This can be a color or a texture or whatever...
            videoCanvasCtx.fillStyle = '#00FF007F';
            videoCanvasCtx.fillRect(0, 0, videoCanvasElement.width, videoCanvasElement.height);
        } else {
            videoCanvasCtx.globalCompositeOperation = 'source-out';
            videoCanvasCtx.fillStyle = '#0000FF7F';
            videoCanvasCtx.fillRect(0, 0, videoCanvasElement.width, videoCanvasElement.height);
        }

        // Only overwrite missing pixels.
        videoCanvasCtx.globalCompositeOperation = 'destination-atop';
        if (results.image)
            videoCanvasCtx.drawImage(
                results.image, 0, 0, videoCanvasElement.width, videoCanvasElement.height);

        videoCanvasCtx.globalCompositeOperation = 'source-over';
    } else {
        if (results.image)
            videoCanvasCtx.drawImage(
                results.image, 0, 0, videoCanvasElement.width, videoCanvasElement.height);
    }

    // Connect elbows to hands. Do this first so that the other graphics will draw
    // on top of these marks.
    videoCanvasCtx.lineWidth = 5;
    if (!!results.poseLandmarks) {
        if (results.rightHandLandmarks) {
            videoCanvasCtx.strokeStyle = 'white';
            connect(videoCanvasCtx, [[
                results.poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW],
                results.rightHandLandmarks[0]
            ]]);
        }
        if (results.leftHandLandmarks) {
            videoCanvasCtx.strokeStyle = 'white';
            connect(videoCanvasCtx, [[
                results.poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW],
                results.leftHandLandmarks[0]
            ]]);
        }

        // Pose...
        drawConnectors(
            videoCanvasCtx, results.poseLandmarks, POSE_CONNECTIONS,
            {color: 'white'});
        drawLandmarks(
            videoCanvasCtx,
            Object.values(POSE_LANDMARKS_LEFT)
                .map(index => results.poseLandmarks[index]),
            {visibilityMin: 0.55, color: 'white', fillColor: 'rgb(255,138,0)'});
        drawLandmarks(
            videoCanvasCtx,
            Object.values(POSE_LANDMARKS_RIGHT)
                .map(index => results.poseLandmarks[index]),
            {visibilityMin: 0.55, color: 'white', fillColor: 'rgb(0,217,231)'});

        // Hands...
        drawConnectors(
            videoCanvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS,
            {color: 'white'});
        drawLandmarks(videoCanvasCtx, results.rightHandLandmarks, {
            color: 'white',
            fillColor: 'rgb(0,217,231)',
            lineWidth: 2,
            radius: (data: Data) => {
                return lerp(data.from!.z!, -0.15, .1, 10, 1);
            }
        });
        drawConnectors(
            videoCanvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS,
            {color: 'white'});
        drawLandmarks(videoCanvasCtx, results.leftHandLandmarks, {
            color: 'white',
            fillColor: 'rgb(255,138,0)',
            lineWidth: 2,
            radius: (data: Data) => {
                return lerp(data.from!.z!, -0.15, .1, 10, 1);
            }
        });

        // Face...
        // drawConnectors(
        //     videoCanvasCtx, results.faceLandmarks, FACEMESH_TESSELATION,
        //     {color: '#C0C0C070', lineWidth: 1});
        drawConnectors(
            videoCanvasCtx, results.faceLandmarks, FACEMESH_RIGHT_IRIS,
            {color: 'rgb(0,217,231)'});
        drawConnectors(
            videoCanvasCtx, results.faceLandmarks, FACEMESH_RIGHT_EYE,
            {color: 'rgb(0,217,231)'});
        drawConnectors(
            videoCanvasCtx, results.faceLandmarks, FACEMESH_RIGHT_EYEBROW,
            {color: 'rgb(0,217,231)'});
        drawConnectors(
            videoCanvasCtx, results.faceLandmarks, FACEMESH_LEFT_IRIS,
            {color: 'rgb(255,138,0)'});
        drawConnectors(
            videoCanvasCtx, results.faceLandmarks, FACEMESH_LEFT_EYE,
            {color: 'rgb(255,138,0)'});
        drawConnectors(
            videoCanvasCtx, results.faceLandmarks, FACEMESH_LEFT_EYEBROW,
            {color: 'rgb(255,138,0)'});
        drawConnectors(
            videoCanvasCtx, results.faceLandmarks, FACEMESH_FACE_OVAL,
            {color: '#E0E0E0', lineWidth: 5});
        drawConnectors(
            videoCanvasCtx, results.faceLandmarks, FACEMESH_LIPS,
            {color: '#E0E0E0', lineWidth: 5});
    }

    videoCanvasCtx.restore();
}

function setHolisticOptions(x: OptionMap, videoElement: HTMLVideoElement, activeEffect: string, holistic: Holistic) {
    const options = x as Options;
    videoElement.classList.toggle('selfie', options.selfieMode);
    // @ts-ignore
    if (options.cameraOn) {
        videoElement.play();
    } else {
        videoElement.pause();
    }
    activeEffect = (x as { [key: string]: string })['effect'];
    holistic.setOptions(options);
}

export function createControlPanel(
    holistic: Holistic,
    videoElement: HTMLVideoElement,
    controlsElement: HTMLDivElement,
    activeEffect: string,
    fpsControl: Nullable<FPS>) {
    setHolisticOptions(holisticOptions, videoElement, activeEffect, holistic);
    if (!controlsElement || !fpsControl) return;

    new ControlPanel(controlsElement, holisticOptions)
        .add([
            new StaticText({title: 'Options'}),
            fpsControl,
            new Toggle({title: 'Selfie Mode', field: 'selfieMode'}),
            new Slider({
                title: 'Model Complexity',
                field: 'modelComplexity',
                discrete: ['Lite', 'Full', 'Heavy'],
            }),
            new Toggle(
                {title: 'Use CPU Only', field: 'useCpuInference'}),
            new Toggle(
                {title: 'Camera on', field: 'cameraOn'}),
            new Toggle(
                {title: 'Smooth Landmarks', field: 'smoothLandmarks'}),
            new Toggle(
                {title: 'Enable Segmentation', field: 'enableSegmentation'}),
            new Toggle(
                {title: 'Smooth Segmentation', field: 'smoothSegmentation'}),
            new Toggle(
                {title: 'Refine Face Landmarks', field: 'refineFaceLandmarks'}),
            new Slider({
                title: 'Min Detection Confidence',
                field: 'minDetectionConfidence',
                range: [0, 1],
                step: 0.01
            }),
            new Slider({
                title: 'Min Tracking Confidence',
                field: 'minTrackingConfidence',
                range: [0, 1],
                step: 0.01
            }),
            new Slider({
                title: 'Effect',
                field: 'effect',
                discrete: {'background': 'Background', 'mask': 'Foreground'},
            }),
        ])
        .on((x) => {
            setHolisticOptions(x, videoElement, activeEffect, holistic);
        });
}
