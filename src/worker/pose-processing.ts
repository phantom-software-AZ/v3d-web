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
    FACEMESH_FACE_OVAL,
    FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_LEFT_IRIS, FACEMESH_LIPS,
    FACEMESH_RIGHT_EYE, FACEMESH_RIGHT_EYEBROW, FACEMESH_RIGHT_IRIS,
    NormalizedLandmark,
    NormalizedLandmarkList,
    POSE_LANDMARKS,
    Results
} from "@mediapipe/holistic";
import {Nullable} from "@babylonjs/core";
import {Vector3} from "@babylonjs/core";
import {
    FACE_LANDMARK_LENGTH,
    GaussianVectorFilter,
    initArray,
    normalizedLandmarkToVector,
    OneEuroVectorFilter,
    POSE_LANDMARK_LENGTH, vectorToNormalizedLandmark
} from "../helper/utils";

type FilteredVectorLandmark3 = [FilteredVectorLandmark, FilteredVectorLandmark, FilteredVectorLandmark];
export interface CloneableResults extends Omit<Results, 'segmentationMask'|'image'> {}

export class Poses {
    public static readonly VISIBILITY_THRESHOLD: number = 0.6;

    public cloneableInputResults: Nullable<CloneableResults> = null;
    public inputPoseLandmarks: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        POSE_LANDMARK_LENGTH, () => {
            return {x: 0, y:0, z: 0};
        });
    private poseLandmarks: FilteredVectorLandmarkList = initArray<FilteredVectorLandmark>(
        POSE_LANDMARK_LENGTH, () => {
            return new FilteredVectorLandmark();
        });
    public inputFaceLandmarks: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        FACE_LANDMARK_LENGTH, () => {
            return {x: 0, y:0, z: 0};
        });
    private faceLandmarks: FilteredVectorLandmarkList = initArray<FilteredVectorLandmark>(
        FACE_LANDMARK_LENGTH, () => {
            return new FilteredVectorLandmark();
        });
    // Cannot use Vector3 directly since Comlink RPC erases all methods
    public cloneablePoseLandmarks: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        POSE_LANDMARK_LENGTH, () => {
            return {x: 0, y:0, z: 0};
        });

    private _faceNormals: NormalizedLandmarkList = [];
    get faceNormals(): NormalizedLandmarkList {
        return this._faceNormals;
    }

    private _faceMeshLandmarkIndexList: number[][] = [];
    get faceMeshLandmarkIndexList(): number[][] {
        return this._faceMeshLandmarkIndexList;
    }
    private _faceMeshLandmarkList: NormalizedLandmarkList[] = [];
    get faceMeshLandmarkList(): NormalizedLandmarkList[] {
        return this._faceMeshLandmarkList;
    }

    private midHipBase: Vector3 = Vector3.Zero();
    private firstResult = true;

    constructor() {}

    public process(results: CloneableResults, dt: number) {
        this.cloneableInputResults = results;
        if (!this.cloneableInputResults) return;

        this.preProcessResults(dt);

        // Actual processing
        // Calculate face center
        this.calcFaceNormal();

        // Post processing
        this.postPoseLandmarks();
    }

    private calcFaceNormal() {
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
        const vertices: FilteredVectorLandmark3[] = [
            [left_eye_inner, mouth_left, right_eye_inner],
            [right_eye_inner, left_eye_inner, mouth_right],
            [mouth_left, mouth_right, nose]
        ]

        // Calculate normals
        const reverse = false;
        this._faceNormals.length = 0;
        const normal = vertices.reduce((prev, curr) => {
                const _normal = Poses.normalFromVertices(curr, reverse);
                // this._faceNormals.push(vectorToNormalizedLandmark(_normal));
                return prev.add(_normal);
            }, Vector3.Zero()).normalize();

        // Face Mesh
        const faceMeshConnections = [
            FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYEBROW,
            FACEMESH_LEFT_EYE, FACEMESH_RIGHT_EYE,
            FACEMESH_LEFT_IRIS, FACEMESH_RIGHT_IRIS,
            FACEMESH_LIPS,
        ]
        this._faceMeshLandmarkIndexList.length = 0;
        this._faceMeshLandmarkList.length = 0;
        if (this.cloneableInputResults?.faceLandmarks) {
            for (let i = 0; i < faceMeshConnections.length; ++i) {
                const arr = [];
                const idx = new Set<number>();
                faceMeshConnections[i].forEach((v) => {
                    idx.add(v[0]);
                    idx.add(v[1]);
                });
                const idxArr = Array.from(idx);
                this._faceMeshLandmarkIndexList.push(idxArr);
                for (let j = 0; j < idxArr.length; j++) {
                    arr.push(this.cloneableInputResults.faceLandmarks[idxArr[j]]);
                }
                this._faceMeshLandmarkList.push(arr);
            }
        }

        this._faceNormals.push(vectorToNormalizedLandmark(normal));
    }

    private static normalFromVertices(vertices: FilteredVectorLandmark3, reverse: boolean) {
        if (reverse)
            vertices.reverse();
        const vec = [];
        for (let i = 0; i < 2; ++i) {
            vec.push(vertices[i + 1].pos.subtract(vertices[i].pos));
        }
        return vec[0].cross(vec[1]).normalize();
    }

    private preProcessResults(dt: number) {
        // Preprocessing results
        // Create pose landmark list
        // @ts-ignore
        const inputPoseLandmarks: NormalizedLandmarkList = this.cloneableInputResults.ea;    // Seems to be the new pose_world_landmark
        if (inputPoseLandmarks) {
            if (inputPoseLandmarks.length !== POSE_LANDMARK_LENGTH)
                console.warn(`Pose Landmark list has a length less than ${POSE_LANDMARK_LENGTH}!`);
            // Remember initial offset
            if (this.firstResult &&
                (inputPoseLandmarks[POSE_LANDMARKS.LEFT_HIP].visibility || 0) > Poses.VISIBILITY_THRESHOLD &&
                (inputPoseLandmarks[POSE_LANDMARKS.RIGHT_HIP].visibility || 0) > Poses.VISIBILITY_THRESHOLD
            ) {
                this.midHipBase =
                    normalizedLandmarkToVector(inputPoseLandmarks[POSE_LANDMARKS.LEFT_HIP])
                        .add(normalizedLandmarkToVector(inputPoseLandmarks[POSE_LANDMARKS.RIGHT_HIP]))
                        .scale(0.5);
                this.firstResult = false;
            }

            this.inputPoseLandmarks = this.preProcessLandmarks(
                inputPoseLandmarks, this.poseLandmarks, dt);
        }

        const inputFaceLandmarks = this.cloneableInputResults?.faceLandmarks;    // Seems to be the new pose_world_landmark
        if (inputFaceLandmarks) {
            this.inputFaceLandmarks = this.preProcessLandmarks(
                inputFaceLandmarks, this.faceLandmarks, dt);
        }
    }

    private preProcessLandmarks(
        resultsLandmarks: NormalizedLandmark[],
        filteredLandmarks: FilteredVectorLandmarkList,
        dt: number,
    ) {
        // Reverse Y-axis. Input results use canvas coordinate system.
        resultsLandmarks.map((v) => {
            v.y = -v.y + this.midHipBase.y;
        });
        // Noise filtering
        for (let i = 0; i < resultsLandmarks.length; ++i) {
            filteredLandmarks[i].updatePosition(
                dt,
                normalizedLandmarkToVector(resultsLandmarks[i]),
                resultsLandmarks[i].visibility);
        }
        return resultsLandmarks;
    }

    private postPoseLandmarks() {
        this.cloneablePoseLandmarks.forEach((v, idx) => {
            v.x = this.poseLandmarks[idx].pos.x;
            v.y = this.poseLandmarks[idx].pos.y;
            v.z = this.poseLandmarks[idx].pos.z;
            v.visibility = this.poseLandmarks[idx].visibility;
        })
    }
}

enum LandmarkTimeIncrementMode {
    Universal,
    RealTime,
}

export class FilteredVectorLandmark {
    private oneEuroVectorFilter: OneEuroVectorFilter;
    // private gaussianVectorFilter: GaussianVectorFilter;
    private _tIncrementMode = LandmarkTimeIncrementMode.Universal;

    private _t = 0;
    get t(): number {
        return this._t;
    }
    set t(value: number) {
        this._t = value;
    }

    private _pos = Vector3.Zero();
    get pos(): Vector3 {
        return this._pos;
    }

    public visibility = 0;

    constructor() {
        this.oneEuroVectorFilter = new OneEuroVectorFilter(
            this.t,
            this.pos,
            Vector3.Zero(),
            0.01,
            1);
        // this.gaussianVectorFilter = new GaussianVectorFilter(10, 1);
    }

    public updatePosition(dt: number, pos: Vector3, visibility?: number) {
        if (this._tIncrementMode === LandmarkTimeIncrementMode.Universal)
            this.t += 1;
        else if (this._tIncrementMode === LandmarkTimeIncrementMode.RealTime)
            this.t += dt;    // Assuming 60 fps

        if (visibility && visibility > Poses.VISIBILITY_THRESHOLD) {
            this._pos = this.oneEuroVectorFilter.next(this.t, pos);

            // this.gaussianVectorFilter.push(this._pos);
            // this._pos = this.gaussianVectorFilter.apply();

            this.visibility = visibility;
        }
    }

    public updateFilterParameters(
        min_cutoff?: number,
        beta?: number,
        d_cutoff?: number) {
        if (min_cutoff) this.oneEuroVectorFilter.min_cutoff = min_cutoff;
        if (beta) this.oneEuroVectorFilter.beta = beta;
        // if (d_cutoff) this.oneEuroVectorFilter.d_cutoff = d_cutoff;
    }
}

export type FilteredVectorLandmarkList = FilteredVectorLandmark[];

const poseResults : Poses = new Poses();

function sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

Comlink.expose(poseResults);

export {poseResults};
