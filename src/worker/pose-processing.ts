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
    EuclideanHighPassFilter,
    FACE_LANDMARK_LENGTH,
    GaussianVectorFilter, HAND_LANDMARK_LENGTH, HAND_LANDMARKS,
    initArray,
    normalizedLandmarkToVector,
    OneEuroVectorFilter,
    POSE_LANDMARK_LENGTH, vectorToNormalizedLandmark
} from "../helper/utils";

type FilteredVectorLandmark3 = [FilteredVectorLandmark, FilteredVectorLandmark, FilteredVectorLandmark];
export interface CloneableResults extends Omit<Results, 'segmentationMask'|'image'> {}

export class Poses {
    public static readonly VISIBILITY_THRESHOLD: number = 0.6;
    public static readonly FACE_MESH_CONNECTIONS = [
        FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYEBROW,
        FACEMESH_LEFT_EYE, FACEMESH_RIGHT_EYE,
        FACEMESH_LEFT_IRIS, FACEMESH_RIGHT_IRIS,
        FACEMESH_LIPS,
    ];
    private static readonly HAND_POSITION_SCALING = 0.8;
    private static readonly HAND_HIGH_PASS_THRESHOLD = 0.008;

    // Results
    public cloneableInputResults: Nullable<CloneableResults> = null;

    // Pose Landmarks
    public inputPoseLandmarks: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        POSE_LANDMARK_LENGTH, () => {
            return {x: 0, y:0, z: 0};
        });
    private poseLandmarks: FilteredVectorLandmarkList = initArray<FilteredVectorLandmark>(
        POSE_LANDMARK_LENGTH, () => {
            return new FilteredVectorLandmark(
                0.01, 2, 0.007);
        });
    // Cannot use Vector3 directly since postMessage erases all methods
    public cloneablePoseLandmarks: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        POSE_LANDMARK_LENGTH, () => {
            return {x: 0, y:0, z: 0};
        });

    // Face Mesh Landmarks
    public inputFaceLandmarks: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        FACE_LANDMARK_LENGTH, () => {
            return {x: 0, y:0, z: 0};
        });
    private faceLandmarks: FilteredVectorLandmarkList = initArray<FilteredVectorLandmark>(
        FACE_LANDMARK_LENGTH, () => {
            return new FilteredVectorLandmark(0.02, 10);
        });
    private _faceMeshLandmarkIndexList: number[][] = [];
    get faceMeshLandmarkIndexList(): number[][] {
        return this._faceMeshLandmarkIndexList;
    }
    private _faceMeshLandmarkList: NormalizedLandmarkList[] = [];
    get faceMeshLandmarkList(): NormalizedLandmarkList[] {
        return this._faceMeshLandmarkList;
    }

    // Left Hand Landmarks
    private leftWristOffset: FilteredVectorLandmark =
        new FilteredVectorLandmark(
            0.01, 2, 0.007);
    public inputLeftHandLandmarks: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        HAND_LANDMARK_LENGTH, () => {
            return {x: 0, y:0, z: 0};
        });
    private leftHandLandmarks: FilteredVectorLandmarkList = initArray<FilteredVectorLandmark>(
        HAND_LANDMARK_LENGTH, () => {
            return new FilteredVectorLandmark(
                0.02, 10, Poses.HAND_HIGH_PASS_THRESHOLD);
        });
    public cloneableLeftHandLandmarks: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        HAND_LANDMARK_LENGTH, () => {
            return {x: 0, y:0, z: 0};
        });

    // Right Hand Landmarks
    private rightWristOffset: FilteredVectorLandmark =
        new FilteredVectorLandmark(
            0.01, 2, 0.007);
    public inputRightHandLandmarks: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        HAND_LANDMARK_LENGTH, () => {
            return {x: 0, y:0, z: 0};
        });
    private rightHandLandmarks: FilteredVectorLandmarkList = initArray<FilteredVectorLandmark>(
        HAND_LANDMARK_LENGTH, () => {
            return new FilteredVectorLandmark(
                0.02, 10, Poses.HAND_HIGH_PASS_THRESHOLD);
        });
    public cloneableRightHandLandmarks: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        HAND_LANDMARK_LENGTH, () => {
            return {x: 0, y:0, z: 0};
        });


    // Calculated properties
    private _faceNormals: NormalizedLandmarkList = [];
    get faceNormals(): NormalizedLandmarkList {
        return this._faceNormals;
    }

    private midHipBase: Nullable<Vector3> = null;

    constructor() {}

    public process(results: CloneableResults, dt: number) {
        this.cloneableInputResults = results;
        if (!this.cloneableInputResults) return;

        this.preProcessResults(dt);

        // Actual processing
        // Post filtered landmarks
        this.postLandmarks(this.poseLandmarks, this.cloneablePoseLandmarks);
        this.postFaceLandmarks();
        this.postLandmarks(this.leftHandLandmarks, this.cloneableLeftHandLandmarks);
        this.postLandmarks(this.rightHandLandmarks, this.cloneableRightHandLandmarks);

        // Calculate face center
        this.calcFaceNormal();

        // Post processing
    }

    /*
     * Calculate the face orientation from landmarks.
     * Landmarks from Face Mesh takes precedence.
     */
    private calcFaceNormal() {
        const normal = Vector3.One();
        let left_eye_inner, right_eye_inner, left_eye_outer, right_eye_outer,
            mouth_left, mouth_right;
        if (!this.cloneableInputResults?.faceLandmarks) {
            // Do not use face landmarks from pose. They are inaccurate.
            if (true || !this.cloneableInputResults?.poseLandmarks) return;

            const nose = this.poseLandmarks[POSE_LANDMARKS.NOSE];
            left_eye_inner = this.poseLandmarks[POSE_LANDMARKS.LEFT_EYE_INNER];
            right_eye_inner = this.poseLandmarks[POSE_LANDMARKS.RIGHT_EYE_INNER];
            left_eye_outer = this.poseLandmarks[POSE_LANDMARKS.LEFT_EYE_OUTER];
            right_eye_outer = this.poseLandmarks[POSE_LANDMARKS.RIGHT_EYE_OUTER];
            // @ts-ignore
            mouth_left = this.poseLandmarks[POSE_LANDMARKS.LEFT_RIGHT];    // Mis-named in MediaPipe JS code
            // @ts-ignore
            mouth_right = this.poseLandmarks[POSE_LANDMARKS.RIGHT_LEFT];    // Mis-named in MediaPipe JS code
        } else {
            // Get points from face mesh
            left_eye_inner = this.faceLandmarks[this.faceMeshLandmarkIndexList[2][8]];
            right_eye_inner = this.faceLandmarks[this.faceMeshLandmarkIndexList[3][8]];
            left_eye_outer = this.faceLandmarks[this.faceMeshLandmarkIndexList[2][0]];
            right_eye_outer = this.faceLandmarks[this.faceMeshLandmarkIndexList[3][0]];
            mouth_left = this.faceLandmarks[this.faceMeshLandmarkIndexList[6][10]];
            mouth_right = this.faceLandmarks[this.faceMeshLandmarkIndexList[6][0]];
        }

        const vertices: FilteredVectorLandmark3[] = [
            [left_eye_inner, mouth_left, right_eye_inner],
            [right_eye_inner, left_eye_inner, mouth_right],
            [mouth_left, mouth_right, right_eye_inner],
            [mouth_left, mouth_right, left_eye_inner],
        ]

        // Calculate normals
        const reverse = false;
        this._faceNormals.length = 0;
        normal.copyFrom(vertices.reduce((prev, curr) => {
            const _normal = Poses.normalFromVertices(curr, reverse);
            // this._faceNormals.push(vectorToNormalizedLandmark(_normal));
            return prev.add(_normal);
        }, Vector3.Zero()).normalize());

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
            if (!this.midHipBase &&
                (inputPoseLandmarks[POSE_LANDMARKS.LEFT_HIP].visibility || 0) > Poses.VISIBILITY_THRESHOLD &&
                (inputPoseLandmarks[POSE_LANDMARKS.RIGHT_HIP].visibility || 0) > Poses.VISIBILITY_THRESHOLD
            ) {
                this.midHipBase =
                    normalizedLandmarkToVector(inputPoseLandmarks[POSE_LANDMARKS.LEFT_HIP])
                        .add(normalizedLandmarkToVector(inputPoseLandmarks[POSE_LANDMARKS.RIGHT_HIP]))
                        .scale(0.5);
            }

            this.inputPoseLandmarks = this.preProcessLandmarks(
                inputPoseLandmarks, this.poseLandmarks, dt);
        }

        const inputFaceLandmarks = this.cloneableInputResults?.faceLandmarks;    // Seems to be the new pose_world_landmark
        if (inputFaceLandmarks) {
            this.inputFaceLandmarks = this.preProcessLandmarks(
                inputFaceLandmarks, this.faceLandmarks, dt);
        }

        const inputLeftHandLandmarks = this.cloneableInputResults?.leftHandLandmarks;
        const inputRightHandLandmarks = this.cloneableInputResults?.rightHandLandmarks;
        if (inputLeftHandLandmarks) {
            // console.log(
            //     this.poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].pos,
            //     normalizedLandmarkToVector(inputLeftHandLandmarks[HAND_LANDMARKS.WRIST]),
            //     this.leftWristOffset.pos
            // );
            this.leftWristOffset.updatePosition(
                dt,
                this.poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].pos.subtract(
                    normalizedLandmarkToVector(
                        inputLeftHandLandmarks[HAND_LANDMARKS.WRIST],
                        Poses.HAND_POSITION_SCALING,
                        true)
                )
            );
            this.inputLeftHandLandmarks = this.preProcessLandmarks(
                inputLeftHandLandmarks, this.leftHandLandmarks, dt,
                this.leftWristOffset.pos, Poses.HAND_POSITION_SCALING);
        }
        if (inputRightHandLandmarks) {
            this.rightWristOffset.updatePosition(
                dt,
                this.poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].pos.subtract(
                    normalizedLandmarkToVector(
                        inputRightHandLandmarks[HAND_LANDMARKS.WRIST],
                        Poses.HAND_POSITION_SCALING,
                        true)
                )
            );
            this.inputRightHandLandmarks = this.preProcessLandmarks(
                inputRightHandLandmarks, this.rightHandLandmarks, dt,
                this.rightWristOffset.pos, Poses.HAND_POSITION_SCALING);
        }
    }

    private preProcessLandmarks(
        resultsLandmarks: NormalizedLandmark[],
        filteredLandmarks: FilteredVectorLandmarkList,
        dt: number,
        offset = Vector3.Zero(),
        scaling = 1.
    ) {
        // Reverse Y-axis. Input results use canvas coordinate system.
        resultsLandmarks.map((v) => {
            v.x = v.x * scaling + offset.x;
            v.y = -v.y * scaling + offset.y;
            v.z = v.z * scaling + offset.z;
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

    private postLandmarks(
        landmarks: FilteredVectorLandmarkList,
        cloneableLandmarks: NormalizedLandmarkList
    ) {
        cloneableLandmarks.forEach((v, idx) => {
            v.x = landmarks[idx].pos.x;
            v.y = landmarks[idx].pos.y;
            v.z = landmarks[idx].pos.z;
            v.visibility = landmarks[idx].visibility;
        })
    }

    private postFaceLandmarks() {
        // Unpack face mesh landmarks
        this._faceMeshLandmarkIndexList.length = 0;
        this._faceMeshLandmarkList.length = 0;
        for (let i = 0; i < Poses.FACE_MESH_CONNECTIONS.length; ++i) {
            const arr = [];
            const idx = new Set<number>();
            Poses.FACE_MESH_CONNECTIONS[i].forEach((v) => {
                idx.add(v[0]);
                idx.add(v[1]);
            });
            const idxArr = Array.from(idx);
            this._faceMeshLandmarkIndexList.push(idxArr);
            for (let j = 0; j < idxArr.length; j++) {
                arr.push({
                    x: this.faceLandmarks[idxArr[j]].pos.x,
                    y: this.faceLandmarks[idxArr[j]].pos.y,
                    z: this.faceLandmarks[idxArr[j]].pos.x,
                    visibility: this.faceLandmarks[idxArr[j]].visibility,
                });
            }
            this._faceMeshLandmarkList.push(arr);
        }
    }
}

enum LandmarkTimeIncrementMode {
    Universal,
    RealTime,
}

export class FilteredVectorLandmark {
    private oneEuroVectorFilter: OneEuroVectorFilter;
    // private gaussianVectorFilter: GaussianVectorFilter;
    private highPassFilter: EuclideanHighPassFilter;
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

    public visibility : number | undefined = 0;

    constructor(
        oneEuroCutoff = 0.01,
        oneEuroBeta = 0,
        highPassThreshold = 0.003,
    ) {
        this.oneEuroVectorFilter = new OneEuroVectorFilter(
            this.t,
            this.pos,
            Vector3.Zero(),
            oneEuroCutoff,
            oneEuroBeta);
        // this.gaussianVectorFilter = new GaussianVectorFilter(10, 1);
        this.highPassFilter = new EuclideanHighPassFilter(highPassThreshold);
    }

    public updatePosition(dt: number, pos: Vector3, visibility?: number) {
        if (this._tIncrementMode === LandmarkTimeIncrementMode.Universal)
            this.t += 1;
        else if (this._tIncrementMode === LandmarkTimeIncrementMode.RealTime)
            this.t += dt;    // Assuming 60 fps

        // Face Mesh has no visibility
        if (!visibility || visibility > Poses.VISIBILITY_THRESHOLD) {
            this._pos = this.oneEuroVectorFilter.next(this.t, pos);

            // this.gaussianVectorFilter.push(this._pos);
            // this._pos = this.gaussianVectorFilter.apply();

            this.highPassFilter.update(this._pos);
            this._pos = this.highPassFilter.value;

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
