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
    FACEMESH_RIGHT_EYE,
    NormalizedLandmark,
    NormalizedLandmarkList,
    POSE_LANDMARKS,
    Results
} from "@mediapipe/holistic";
import {Nullable} from "@babylonjs/core";
import {Vector3} from "@babylonjs/core";
import {initArray, POSE_LANDMARK_LENGTH} from "../helper/utils";

type VectorizedLandmark3 = [VectorizedLandmark, VectorizedLandmark, VectorizedLandmark];
export interface CloneableResults extends Omit<Results, 'segmentationMask'|'image'> {}

export class Poses {
    public cloneableResults: Nullable<CloneableResults> = null;
    private poseLandmarks: VectorizedLandmarkList = initArray<VectorizedLandmark>(
        POSE_LANDMARK_LENGTH, () => {
            return {pos: Vector3.Zero()}
        });
    // Cannot use Vector3 directly since Comlink RPC erases all methods
    public cloneablePoseLandmarks: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        POSE_LANDMARK_LENGTH, () => {
            return {x: 0, y:0, z: 0};
        });

    private _faceNormal: Vector3 = Vector3.Zero();
    get faceNormal(): Vector3 {
        return this._faceNormal;
    }
    set faceNormal(value: Vector3) {
        this._faceNormal = value;
    }

    constructor() {}

    public process (results: CloneableResults) {
        this.cloneableResults = results;
        if (!this.cloneableResults) return;

        // Calculate face center
        this.calcFaceNormal(this.cloneableResults);
        console.log(this._faceNormal);

        // Create pose landmark list
        if (results.poseLandmarks) {
            if (results.poseLandmarks.length != POSE_LANDMARK_LENGTH)
                console.warn(`Pose Landmark list has a length less than ${POSE_LANDMARK_LENGTH}!`);
            this.cloneablePoseLandmarks = results.poseLandmarks;
            this.poseLandmarks = results.poseLandmarks.map((v) => {
                return {pos: normalizedLandmarkToVector(v), visibility: v.visibility};
            });
        }
    }

    private calcFaceNormal(results: CloneableResults) {
        if (!this.poseLandmarks)
            return;
        const nose = this.poseLandmarks[POSE_LANDMARKS.NOSE];
        const left_eye_inner = this.poseLandmarks[POSE_LANDMARKS.LEFT_EYE_INNER];
        const right_eye_inner = this.poseLandmarks[POSE_LANDMARKS.RIGHT_EYE_INNER];
        const left_eye_outer = this.poseLandmarks[POSE_LANDMARKS.LEFT_EYE_OUTER];
        const right_eye_outer = this.poseLandmarks[POSE_LANDMARKS.RIGHT_EYE_OUTER];
        // @ts-ignore
        const mouth_left = this.poseLandmarks[POSE_LANDMARKS.LEFT_RIGHT];    // Mis-named in MediaPipe JS code
        // @ts-ignore
        const mouth_right = this.poseLandmarks[POSE_LANDMARKS.RIGHT_LEFT];    // Mis-named in MediaPipe JS code
        const vertices: VectorizedLandmark3[] = [
            [left_eye_inner, left_eye_outer, nose],
            [right_eye_outer, right_eye_inner, nose],
            [mouth_left, mouth_right, nose]
        ]

        // Calculate normals
        const reverse = true;
        const normal = vertices.reduce((prev, curr) =>
                prev.add(Poses.normalFromVertices(curr, reverse)),
            Vector3.Zero()).normalize();

        // TODO: use face mesh instead
        // Debug
        if (results.faceLandmarks) {
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
        }

        this._faceNormal = normal;
    }

    private static normalFromVertices(vertices: VectorizedLandmark3, reverse: boolean) {
        if (reverse)
            vertices.reverse();
        const vec = [];
        for (let i = 0; i < 2; ++i) {
            vec.push(vertices[i + 1].pos.subtract(vertices[i].pos));
        }
        return vec[0].cross(vec[1]).normalize();
    }

    // Debug
    public counter = 0;
    public inc(i: number = 1) {
        this.counter += i;
    }
    public async spin (){
        let i = 0;
        while (i < 10000) {
            console.log(i);
            await sleep(i);
            i++;
        }
    }
}

export declare interface VectorizedLandmark {
    pos: Vector3,
    visibility?: number;
}

export type VectorizedLandmarkList = VectorizedLandmark[];

const normalizedLandmarkToVector = (l: NormalizedLandmark) => new Vector3(l.x, l.y, l.z);

const poseResults : Poses = new Poses();

function sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

Comlink.expose(poseResults);

export {poseResults};
