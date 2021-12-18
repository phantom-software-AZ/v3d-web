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

import * as Comlink from "comlink"
import {
    FaceGeometry, FACEMESH_RIGHT_EYE,
    GpuBuffer,
    NormalizedLandmark,
    NormalizedLandmarkList,
    POSE_LANDMARKS,
    Results
} from "@mediapipe/holistic";
import {Nullable, Quaternion} from "@babylonjs/core";
import {Vector3} from "@babylonjs/core";

export interface CloneableResults extends Omit<Results, 'segmentationMask'|'image'> {}

export interface Poses {
    results: Nullable<CloneableResults>,
    process: (r: CloneableResults) => void,
    // Debug
    counter: number,
    inc: (i: number) => void,
    spin: () => void,
}

const normalizedLandmarkToVector = (l: NormalizedLandmark) => new Vector3(l.x, l.y, l.z);

function calcFaceCenter(results: CloneableResults): Vector3 {
    const nose = normalizedLandmarkToVector(results.poseLandmarks[POSE_LANDMARKS.NOSE]);
    const left_eye = normalizedLandmarkToVector(results.poseLandmarks[POSE_LANDMARKS.LEFT_EYE]);
    const right_eye = normalizedLandmarkToVector(results.poseLandmarks[POSE_LANDMARKS.RIGHT_EYE]);
    const left_ear = normalizedLandmarkToVector(results.poseLandmarks[POSE_LANDMARKS.LEFT_EAR]);
    const right_ear = normalizedLandmarkToVector(results.poseLandmarks[POSE_LANDMARKS.RIGHT_EAR]);
    console.log(right_eye);
    // Mid points
    const mid_eye = left_eye.add(right_eye).scaleInPlace(0.5);
    const mid_ear = left_ear.add(right_ear).scaleInPlace(0.5);

    const res = Vector3.Zero();
    const arr = [];
    const idx = new Set<number>();
    FACEMESH_RIGHT_EYE.forEach((v) => {
        idx.add(v[0]);
        idx.add(v[1]);
    });
    const idxArr = Array.from(idx);
    for (let i = 0; i <= idxArr.length; i++) {
        arr.push(results.faceLandmarks[idxArr[i]]);
    }
    console.log(arr);
    // mid_ear.subtract(mid_eye).cross(mid_ear.subtract(nose)).rotateByQuaternionToRef(
    //     Quaternion.FromEulerAngles(0, mesh.rotation.y, 0), res);

    return null;
}

const poseResults : Poses = {
    results: null,
    process: function (results) {
        this.results = results;
        if (!this.results) return;
        // console.log(this.results);

        // Calculate face center
        const face_center = calcFaceCenter(this.results);
    },
    // Debug
    counter: 0,
    inc(i: number = 1) {
        this.counter += i;
    },
    async spin() {
        let i = 0;
        while (i < 10000) {
            console.log(i);
            await sleep(i);
            i++;
        }
    }
}

function sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

Comlink.expose(poseResults);

export {poseResults};
