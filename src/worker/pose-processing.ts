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
import {Matrix, Nullable, Quaternion, TransformNode} from "@babylonjs/core";
import {Vector3} from "@babylonjs/core";
import {
    axesToBasis,
    AXIS, calcAvgPlane,
    degreeBetweenVectors,
    EuclideanHighPassFilter,
    FACE_LANDMARK_LENGTH,
    GaussianVectorFilter, getAxes,
    HAND_BONE_NODES,
    HAND_LANDMARK_LENGTH,
    HAND_LANDMARKS,
    initArray,
    KeysMatching,
    NodeWorldMatrixMap,
    normalizedLandmarkToVector,
    OneEuroVectorFilter,
    POSE_LANDMARK_LENGTH, printQuaternion, quaternionBetweenBases,
    quaternionBetweenVectors,
    quaternionToDegrees,
    ReadonlyKeys,
    remapRangeWithCap, reverseRotation, Vector33,
    vectorToNormalizedLandmark
} from "../helper/utils";

type FilteredVectorLandmark3 = [
    FilteredVectorLandmark,
    FilteredVectorLandmark,
    FilteredVectorLandmark,
];
export interface CloneableResults extends Omit<Results, 'segmentationMask'|'image'> {}
export class CloneableQuaternion {
    public x: number = 0;
    public y: number = 0;
    public z: number = 0;
    public w: number = 1;

    constructor(q: Quaternion) {
        this.set(q);
    }

    public set(q: Quaternion) {
        this.x = q.x;
        this.y = q.y;
        this.z = q.z;
        this.w = q.w;
    }
}
export type CloneableQuaternionList = CloneableQuaternion[];
export class PoseKeyPoints {
    public nose = new FilteredVectorLandmark();
    public left_eye_top = new FilteredVectorLandmark();
    public left_eye_bottom = new FilteredVectorLandmark();
    public left_eye_inner = new FilteredVectorLandmark();
    public left_eye_outer = new FilteredVectorLandmark();
    public left_eye_inner_secondary = new FilteredVectorLandmark();
    public left_eye_outer_secondary = new FilteredVectorLandmark();
    public left_iris_top = new FilteredVectorLandmark();
    public left_iris_bottom = new FilteredVectorLandmark();
    public left_iris_left = new FilteredVectorLandmark();
    public left_iris_right = new FilteredVectorLandmark();
    public right_eye_top = new FilteredVectorLandmark();
    public right_eye_bottom = new FilteredVectorLandmark();
    public right_eye_inner = new FilteredVectorLandmark();
    public right_eye_outer = new FilteredVectorLandmark();
    public right_eye_inner_secondary = new FilteredVectorLandmark();
    public right_eye_outer_secondary = new FilteredVectorLandmark();
    public right_iris_top = new FilteredVectorLandmark();
    public right_iris_bottom = new FilteredVectorLandmark();
    public right_iris_left = new FilteredVectorLandmark();
    public right_iris_right = new FilteredVectorLandmark();
    public mouth_top_first = new FilteredVectorLandmark();
    public mouth_top_second = new FilteredVectorLandmark();
    public mouth_top_third = new FilteredVectorLandmark();
    public mouth_bottom_first = new FilteredVectorLandmark();
    public mouth_bottom_second = new FilteredVectorLandmark();
    public mouth_bottom_third = new FilteredVectorLandmark();
    public mouth_left = new FilteredVectorLandmark();
    public mouth_right = new FilteredVectorLandmark();
}

export type PosesKeys = keyof Omit<Poses, KeysMatching<Poses, Function> | ReadonlyKeys<Poses>>;

export class Poses {
    public static readonly VISIBILITY_THRESHOLD: number = 0.6;
    public static readonly FACE_MESH_CONNECTIONS = [
        FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYEBROW,
        FACEMESH_LEFT_EYE, FACEMESH_RIGHT_EYE,
        FACEMESH_LEFT_IRIS, FACEMESH_RIGHT_IRIS,
        FACEMESH_LIPS,
    ];
    public static readonly HAND_BASE_ROOT_NORMAL = new Vector3(0, -1, 0);


    private static readonly HAND_POSITION_SCALING = 0.8;
    private static readonly HAND_HIGH_PASS_THRESHOLD = 0.008;

    private static readonly IRIS_MP_X_RANGE = 0.027;
    private static readonly IRIS_MP_Y_RANGE = 0.011;
    private static readonly IRIS_BJS_X_RANGE = 0.28;
    private static readonly IRIS_BJS_Y_RANGE = 0.22;

    private static readonly BLINK_MP_RANGE_LOW = 0.0155;
    private static readonly BLINK_MP_RANGE_HIGH = 0.018;
    private static readonly MOUTH_MP_RANGE_LOW = 0.0006;
    private static readonly MOUTH_MP_RANGE_HIGH = 0.06;

    private static readonly EYE_WIDTH_BASELINE = 0.0526;
    private static readonly MOUTH_WIDTH_BASELINE = 0.095;
    private static readonly LR_FACE_DIRECTION_RANGE = 27;

    // VRMManager
    private leftHandBoneWorldMatrices: Nullable<NodeWorldMatrixMap> = null;
    private rightHandBoneWorldMatrices: Nullable<NodeWorldMatrixMap> = null;

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
    // Cannot use Vector3 directly since postMessage() erases all methods
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
            return new FilteredVectorLandmark(
                0.001, 15, 0.002);
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
                0.0001, 10,
                Poses.HAND_HIGH_PASS_THRESHOLD,
                5);
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
                0.0001, 10,
                Poses.HAND_HIGH_PASS_THRESHOLD,
                5);
        });
    public cloneableRightHandLandmarks: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        HAND_LANDMARK_LENGTH, () => {
            return {x: 0, y:0, z: 0};
        });

    // Key points
    private _keyPoints: PoseKeyPoints = new PoseKeyPoints();
    get keyPoints(): PoseKeyPoints {
        return this._keyPoints;
    }

    // Calculated properties
    private _faceNormal: NormalizedLandmark = {x: 0, y: 0, z: 0};
    get faceNormal(): NormalizedLandmark {
        return this._faceNormal;
    }
    private _irisQuaternion: CloneableQuaternionList = [
        new CloneableQuaternion(new Quaternion(0, 0, 0, 0)),
        new CloneableQuaternion(new Quaternion(0, 0, 0, 0)),
        new CloneableQuaternion(new Quaternion(0, 0, 0, 0)),
    ];
    get irisQuaternion(): CloneableQuaternionList {
        return this._irisQuaternion;
    }
    private _mouthMorph: number = 0;
    get mouthMorph(): number {
        return this._mouthMorph;
    }
    private _blinkLeft: number = 1;
    get blinkLeft(): number {
        return this._blinkLeft;
    }
    private _blinkRight: number = 1;
    get blinkRight(): number {
        return this._blinkRight;
    }
    private _blinkAll: number = 1;
    get blinkAll(): number {
        return this._blinkAll;
    }

    // Calculated bone rotations
    private _leftHandBoneRotations: CloneableQuaternionList = initArray<CloneableQuaternion>(
        HAND_LANDMARK_LENGTH, () => {
            return new CloneableQuaternion(
                new Quaternion(0, 0, 0, 0));
        });
    get leftHandBoneRotations(): CloneableQuaternionList {
        return this._leftHandBoneRotations;
    }
    private _leftHandNormals: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        3, () => {
            return {x: 0, y:0, z: 0};
        });
    get leftHandNormals(): NormalizedLandmarkList {
        return this._leftHandNormals;
    }
    private _rightHandBoneRotations: CloneableQuaternionList = initArray<CloneableQuaternion>(
        HAND_LANDMARK_LENGTH, () => {
            return new CloneableQuaternion(
                new Quaternion(0, 0, 0, 0));
        });
    get rightHandBoneRotations(): CloneableQuaternionList {
        return this._rightHandBoneRotations;
    }
    private _rightHandNormals: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        3, () => {
            return {x: 0, y:0, z: 0};
        });
    get rightHandNormals(): NormalizedLandmarkList {
        return this._rightHandNormals;
    }

    private midHipBase: Nullable<Vector3> = null;

    constructor() {}

    public bindHumanoidWorldMatrix(boneMap: NodeWorldMatrixMap) {
        this.leftHandBoneWorldMatrices = {};
        this.rightHandBoneWorldMatrices = {};
        for (const k in boneMap) {
            if (k.includes('Proximal') || k.includes('Intermediate')
                || k.includes('Distal') ||  k.includes('Hand')) {
                if (k[0] === 'l')
                    this.leftHandBoneWorldMatrices[k] = boneMap[k];
                else
                    this.rightHandBoneWorldMatrices[k] = boneMap[k];
            }
        }
    }

    public process(results: CloneableResults, dt: number) {
        this.cloneableInputResults = results;
        if (!this.cloneableInputResults) return;

        this.preProcessResults(dt);

        // Actual processing
        // Post filtered landmarks
        this.toCloneableLandmarks(this.poseLandmarks, this.cloneablePoseLandmarks);
        this.filterFaceLandmarks();
        this.toCloneableLandmarks(this.leftHandLandmarks, this.cloneableLeftHandLandmarks);
        this.toCloneableLandmarks(this.rightHandLandmarks, this.cloneableRightHandLandmarks);

        // Gather key points
        this.getKeyPoints();

        // Calculate face orientation
        this.calcFaceNormal();

        // Calculate iris orientations
        this.calcIrisNormal();

        // Calculate expressions
        this.calcExpressions();

        // Calculate hand bones
        this.calcHandBones();

        // Post processing
    }

    private getKeyPoints() {
        if (!this.cloneableInputResults?.faceLandmarks) {
            // Do not use face landmarks from pose. They are inaccurate.
            if (true || !this.cloneableInputResults?.poseLandmarks) return;

            this._keyPoints.nose = this.poseLandmarks[POSE_LANDMARKS.NOSE];
            this._keyPoints.left_eye_inner = this.poseLandmarks[POSE_LANDMARKS.LEFT_EYE_INNER];
            this._keyPoints.right_eye_inner = this.poseLandmarks[POSE_LANDMARKS.RIGHT_EYE_INNER];
            this._keyPoints.left_eye_outer = this.poseLandmarks[POSE_LANDMARKS.LEFT_EYE_OUTER];
            this._keyPoints.right_eye_outer = this.poseLandmarks[POSE_LANDMARKS.RIGHT_EYE_OUTER];
            // @ts-ignore
            this._keyPoints.mouth_left = this.poseLandmarks[POSE_LANDMARKS.LEFT_RIGHT];    // Mis-named in MediaPipe JS code
            // @ts-ignore
            this._keyPoints.mouth_right = this.poseLandmarks[POSE_LANDMARKS.RIGHT_LEFT];    // Mis-named in MediaPipe JS code
        } else {
            // Get points from face mesh
            this._keyPoints.left_eye_inner = this.faceLandmarks[this.faceMeshLandmarkIndexList[2][8]];
            this._keyPoints.right_eye_inner = this.faceLandmarks[this.faceMeshLandmarkIndexList[3][8]];
            this._keyPoints.left_eye_outer = this.faceLandmarks[this.faceMeshLandmarkIndexList[2][0]];
            this._keyPoints.right_eye_outer = this.faceLandmarks[this.faceMeshLandmarkIndexList[3][0]];

            this._keyPoints.mouth_left = this.faceLandmarks[this.faceMeshLandmarkIndexList[6][10]];
            this._keyPoints.mouth_right = this.faceLandmarks[this.faceMeshLandmarkIndexList[6][0]];
            this._keyPoints.mouth_top_first = this.faceLandmarks[this.faceMeshLandmarkIndexList[6][24]];
            this._keyPoints.mouth_top_second = this.faceLandmarks[this.faceMeshLandmarkIndexList[6][25]];
            this._keyPoints.mouth_top_third = this.faceLandmarks[this.faceMeshLandmarkIndexList[6][26]];
            this._keyPoints.mouth_bottom_first = this.faceLandmarks[this.faceMeshLandmarkIndexList[6][34]];
            this._keyPoints.mouth_bottom_second = this.faceLandmarks[this.faceMeshLandmarkIndexList[6][35]];
            this._keyPoints.mouth_bottom_third = this.faceLandmarks[this.faceMeshLandmarkIndexList[6][36]];

            this._keyPoints.left_iris_top = this.faceLandmarks[this.faceMeshLandmarkIndexList[4][1]];
            this._keyPoints.left_iris_bottom = this.faceLandmarks[this.faceMeshLandmarkIndexList[4][3]];
            this._keyPoints.left_iris_left = this.faceLandmarks[this.faceMeshLandmarkIndexList[4][2]];
            this._keyPoints.left_iris_right = this.faceLandmarks[this.faceMeshLandmarkIndexList[4][0]];
            this._keyPoints.right_iris_top = this.faceLandmarks[this.faceMeshLandmarkIndexList[5][1]];
            this._keyPoints.right_iris_bottom = this.faceLandmarks[this.faceMeshLandmarkIndexList[5][3]];
            this._keyPoints.right_iris_left = this.faceLandmarks[this.faceMeshLandmarkIndexList[5][2]];
            this._keyPoints.right_iris_right = this.faceLandmarks[this.faceMeshLandmarkIndexList[5][0]];

            this._keyPoints.left_eye_top = this.faceLandmarks[this.faceMeshLandmarkIndexList[2][12]];
            this._keyPoints.left_eye_bottom = this.faceLandmarks[this.faceMeshLandmarkIndexList[2][4]];
            this._keyPoints.left_eye_inner_secondary = this.faceLandmarks[this.faceMeshLandmarkIndexList[2][14]];
            this._keyPoints.left_eye_outer_secondary = this.faceLandmarks[this.faceMeshLandmarkIndexList[2][10]];
            this._keyPoints.right_eye_top = this.faceLandmarks[this.faceMeshLandmarkIndexList[3][12]];
            this._keyPoints.right_eye_bottom = this.faceLandmarks[this.faceMeshLandmarkIndexList[3][4]];
            this._keyPoints.right_eye_outer_secondary = this.faceLandmarks[this.faceMeshLandmarkIndexList[3][10]];
            this._keyPoints.right_eye_inner_secondary = this.faceLandmarks[this.faceMeshLandmarkIndexList[3][14]];
        }
    }

    /*
     * Calculate the face orientation from landmarks.
     * Landmarks from Face Mesh takes precedence.
     */
    private calcFaceNormal() {
        const normal = Vector3.One();

        const vertices: FilteredVectorLandmark3[] = [
            [this._keyPoints.left_eye_inner, this._keyPoints.mouth_left, this._keyPoints.right_eye_inner],
            [this._keyPoints.right_eye_inner, this._keyPoints.left_eye_inner, this._keyPoints.mouth_right],
            [this._keyPoints.mouth_left, this._keyPoints.mouth_right, this._keyPoints.right_eye_inner],
            [this._keyPoints.mouth_left, this._keyPoints.mouth_right, this._keyPoints.left_eye_inner],
        ]

        // Calculate normals
        const reverse = false;
        normal.copyFrom(vertices.reduce((prev, curr) => {
            const _normal = Poses.normalFromVertices(curr, reverse);
            // this._faceNormals.push(vectorToNormalizedLandmark(_normal));
            return prev.add(_normal);
        }, Vector3.Zero()).normalize());

        this._faceNormal = vectorToNormalizedLandmark(normal);
    }

    /*
     * Remap positional offsets to rotations.
     * Iris only have positional offsets. Their normals always face front.
     */
    private calcIrisNormal() {
        if (!this.cloneableInputResults?.faceLandmarks) return;

        const leftIrisCenter = this._keyPoints.left_iris_top.pos
            .add(this._keyPoints.left_iris_bottom.pos)
            .add(this._keyPoints.left_iris_left.pos)
            .add(this._keyPoints.left_iris_right.pos)
            .scale(0.5);
        const rightIrisCenter = this._keyPoints.right_iris_top.pos
            .add(this._keyPoints.right_iris_bottom.pos)
            .add(this._keyPoints.right_iris_left.pos)
            .add(this._keyPoints.right_iris_right.pos)
            .scale(0.5);

        // Calculate eye center
        const leftEyeCenter = this._keyPoints.left_eye_top.pos
            .add(this._keyPoints.left_eye_bottom.pos)
            .add(this._keyPoints.left_eye_inner_secondary.pos)
            .add(this._keyPoints.left_eye_outer_secondary.pos)
            .scale(0.5);
        const rightEyeCenter = this._keyPoints.right_eye_top.pos
            .add(this._keyPoints.right_eye_bottom.pos)
            .add(this._keyPoints.right_eye_outer_secondary.pos)
            .add(this._keyPoints.right_eye_inner_secondary.pos)
            .scale(0.5);

        // Calculate offsets
        const leftEyeWidth = this._keyPoints.left_eye_inner.pos.subtract(this._keyPoints.left_eye_outer.pos).length();
        const rightEyeWidth = this._keyPoints.right_eye_inner.pos.subtract(this._keyPoints.right_eye_outer.pos).length();

        const leftIrisOffset = leftIrisCenter
            .subtract(leftEyeCenter)
            .scale(Poses.EYE_WIDTH_BASELINE / leftEyeWidth);
        const rightIrisOffset = rightIrisCenter
            .subtract(rightEyeCenter)
            .scale(Poses.EYE_WIDTH_BASELINE / rightEyeWidth);

        /* Remap offsets to quaternions
         * Using arbitrary range:
         * MediaPipe Iris:
         * x: -0.03, 0.03
         * y: -0.025, 0.025
         * BabylonJS RotationYawPitchRoll:
         * x: -0.25, 0.25
         * y: -0.25, 0.25
         */
        const leftIrisRotationYPR = Quaternion.RotationYawPitchRoll(
            remapRangeWithCap(leftIrisOffset.x, -Poses.IRIS_MP_X_RANGE, Poses.IRIS_MP_X_RANGE,
                -Poses.IRIS_BJS_X_RANGE, Poses.IRIS_BJS_X_RANGE),
            remapRangeWithCap(leftIrisOffset.y, -Poses.IRIS_MP_Y_RANGE, Poses.IRIS_MP_Y_RANGE,
                -Poses.IRIS_BJS_Y_RANGE, Poses.IRIS_BJS_Y_RANGE),
            0
        );
        const rightIrisRotationYPR = Quaternion.RotationYawPitchRoll(
            remapRangeWithCap(rightIrisOffset.x, -Poses.IRIS_MP_X_RANGE, Poses.IRIS_MP_X_RANGE,
                -Poses.IRIS_BJS_X_RANGE, Poses.IRIS_BJS_X_RANGE),
            remapRangeWithCap(rightIrisOffset.y, -Poses.IRIS_MP_Y_RANGE, Poses.IRIS_MP_Y_RANGE,
                -Poses.IRIS_BJS_Y_RANGE, Poses.IRIS_BJS_Y_RANGE),
            0
        );

        this._irisQuaternion.length = 0;
        this._irisQuaternion.push(new CloneableQuaternion(leftIrisRotationYPR));
        this._irisQuaternion.push(new CloneableQuaternion(rightIrisRotationYPR));
        this._irisQuaternion.push(new CloneableQuaternion(this.lRLinkQuaternion(
            leftIrisRotationYPR, rightIrisRotationYPR)));
    }

    private calcExpressions() {
        if (!this.cloneableInputResults?.faceLandmarks) return;

        const leftEyeWidth = this._keyPoints.left_eye_inner.pos.subtract(this._keyPoints.left_eye_outer.pos).length();
        this._blinkLeft = 1 - remapRangeWithCap(
            this._keyPoints.left_eye_top.pos
                .subtract(this._keyPoints.left_eye_bottom.pos)
                .length() * Poses.EYE_WIDTH_BASELINE / leftEyeWidth,
            Poses.BLINK_MP_RANGE_LOW, Poses.BLINK_MP_RANGE_HIGH,
            0, 1
        );
        const rightEyeWidth = this._keyPoints.right_eye_inner.pos.subtract(this._keyPoints.right_eye_outer.pos).length();
        this._blinkRight = 1 - remapRangeWithCap(
            this._keyPoints.right_eye_top.pos
                .subtract(this._keyPoints.right_eye_bottom.pos)
                .length() * Poses.EYE_WIDTH_BASELINE / rightEyeWidth,
            Poses.BLINK_MP_RANGE_LOW, Poses.BLINK_MP_RANGE_HIGH,
            0, 1
        );
        this._blinkAll = this.lRLink(this._blinkLeft, this._blinkRight);

        const mouthWidth = this._keyPoints.mouth_left.pos.subtract(this._keyPoints.mouth_right.pos).length();
        const mouthRange1 = remapRangeWithCap(
            this._keyPoints.mouth_top_first.pos.subtract(this._keyPoints.mouth_bottom_first.pos)
                .length() * Poses.MOUTH_WIDTH_BASELINE / mouthWidth,
            Poses.MOUTH_MP_RANGE_LOW, Poses.MOUTH_MP_RANGE_HIGH,
            0, 1
        );
        const mouthRange2 = remapRangeWithCap(
            this._keyPoints.mouth_top_second.pos.subtract(this._keyPoints.mouth_bottom_second.pos)
                .length() * Poses.MOUTH_WIDTH_BASELINE / mouthWidth,
            Poses.MOUTH_MP_RANGE_LOW, Poses.MOUTH_MP_RANGE_HIGH,
            0, 1
        );
        const mouthRange3 = remapRangeWithCap(
            this._keyPoints.mouth_top_third.pos.subtract(this._keyPoints.mouth_bottom_third.pos)
                .length() * Poses.MOUTH_WIDTH_BASELINE / mouthWidth,
            Poses.MOUTH_MP_RANGE_LOW, Poses.MOUTH_MP_RANGE_HIGH,
            0, 1
        );
        this._mouthMorph = (mouthRange1 + mouthRange2 + mouthRange3) / 3;
    }

    private calcHandBones() {
        // Right hand shall have local x reversed?
        const hands = {
            left: this.leftHandLandmarks,
            right: this.rightHandLandmarks,
        }

        for (const [k, v] of Object.entries(hands)) {
            const isLeft = k === 'left';
            const thisHandBoneWorldMatrices = isLeft
                ? this.leftHandBoneWorldMatrices
                : this.rightHandBoneWorldMatrices;

            if (!thisHandBoneWorldMatrices) return;

            const vertices: FilteredVectorLandmark3[] = [
                [v[HAND_LANDMARKS.WRIST], v[HAND_LANDMARKS.PINKY_MCP], v[HAND_LANDMARKS.INDEX_FINGER_MCP]],
                [v[HAND_LANDMARKS.WRIST], v[HAND_LANDMARKS.RING_FINGER_MCP], v[HAND_LANDMARKS.INDEX_FINGER_MCP]],
                [v[HAND_LANDMARKS.WRIST], v[HAND_LANDMARKS.PINKY_MCP], v[HAND_LANDMARKS.MIDDLE_FINGER_MCP]],
            ]

            // Root normal
            const handNormalsKey = `${k}HandNormals`;
            const handNormals = this[handNormalsKey as PosesKeys] as NormalizedLandmarkList;
            handNormals.length = 0;
            const rootNormal = vertices.reduce((prev, curr) => {
                const _normal = Poses.normalFromVertices(curr, isLeft);
                // handNormals.push(vectorToNormalizedLandmark(_normal));
                return prev.add(_normal);
            }, Vector3.Zero()).normalize();
            // handNormals.push(vectorToNormalizedLandmark(rootNormal));

            const handRotationKey = `${k}HandBoneRotations`;
            const handRotations = this[handRotationKey as PosesKeys] as CloneableQuaternionList;

            const axes1: Vector33 = isLeft ? getAxes(
                [
                    new Vector3(0, 0, 0),
                    new Vector3(isLeft ? 1 : -1, 0, 0),
                    new Vector3(isLeft ? 1 : -1, 0, 1)
                ]) : [
                    new Vector3(-0.9327159079568041, 0.12282522615654383, -0.3390501421086685),
                    new Vector3(-0.010002212677077182, 0.0024727643453822945, 0.028411551927747327),
                    new Vector3(0.14320801411112857, 0.9890497926949048, -0.03566472016590984)
                ];
            const basis1 = axesToBasis(axes1);

            // Project palm landmarks to average plane
            const projectedLandmarks = calcAvgPlane([
                v[HAND_LANDMARKS.WRIST].pos,
                v[HAND_LANDMARKS.INDEX_FINGER_MCP].pos,
                v[HAND_LANDMARKS.MIDDLE_FINGER_MCP].pos,
                v[HAND_LANDMARKS.RING_FINGER_MCP].pos,
                v[HAND_LANDMARKS.PINKY_MCP].pos
            ], rootNormal);
            const axes2 = getAxes([
                projectedLandmarks[0],
                projectedLandmarks[1],
                projectedLandmarks[4]
            ]);
            const basis2 = axesToBasis(axes2);
            const wristRotationQuaternionRaw = quaternionBetweenBases(basis1, basis2);

            if (!isLeft) printQuaternion(wristRotationQuaternionRaw, "wristRotationQuaternionRaw: ");
            handNormals.push(vectorToNormalizedLandmark(axes1[0]));
            handNormals.push(vectorToNormalizedLandmark(axes1[1]));
            handNormals.push(vectorToNormalizedLandmark(axes1[2]));
            handNormals.push(vectorToNormalizedLandmark(axes2[0]));
            handNormals.push(vectorToNormalizedLandmark(axes2[1]));
            handNormals.push(vectorToNormalizedLandmark(axes2[2]));

            const wristRotationQuaternion = reverseRotation(wristRotationQuaternionRaw, AXIS.yz);
            handRotations[HAND_LANDMARKS.WRIST].set(wristRotationQuaternion);
            const wristRotationDegrees = quaternionToDegrees(wristRotationQuaternion);

            const baseFingerDir = new Vector3(k === 'left' ? 1 : -1, 0, 0);

            for (let i = 1; i < HAND_LANDMARK_LENGTH; ++i) {
                if (i % 4 === 0) continue;

                const thisLandmark = v[i].pos.clone();
                const nextLandmark = v[i + 1].pos.clone();
                const thisDir = nextLandmark.subtract(thisLandmark);
                // const prevQuaternion = cloneableQuaternionToQuaternion(handRotations[(i - 1) % 4 === 0 ? 0 : i - 1]);
                // const prevRotationMat = Matrix.Identity();
                const invPrevRotationMat = Matrix.Identity();
                const prevBoneKey = `${k}${HAND_BONE_NODES[i - 1]}`;
                // @ts-ignore
                const prevNodeWorldMat = Matrix.FromArray(thisHandBoneWorldMatrices[prevBoneKey]._m);
                const prevRotationMat = prevNodeWorldMat.getRotationMatrix().clone();
                const prevRotationDegrees = quaternionToDegrees(Quaternion.FromRotationMatrix(prevRotationMat));
                // prevQuaternion.toRotationMatrix(prevRotationMat);
                prevRotationMat.invertToRef(invPrevRotationMat);
                const thisRotationQuaternion = quaternionBetweenVectors(thisDir, baseFingerDir);
                const thisRotationDegreesBefore = quaternionToDegrees(thisRotationQuaternion);
                const thisRotationMat = Matrix.Identity();
                thisRotationQuaternion.toRotationMatrix(thisRotationMat);
                const thisQuaternion = Quaternion.FromRotationMatrix(thisRotationMat.multiply(invPrevRotationMat));
                const thisRotationDegreesAfter = quaternionToDegrees(thisQuaternion);
                // if (i === HAND_LANDMARKS.MIDDLE_FINGER_MCP || i === HAND_LANDMARKS.MIDDLE_FINGER_PIP || i === HAND_LANDMARKS.MIDDLE_FINGER_DIP) {
                //     console.debug(`${k}${HAND_BONE_NODES[i]}`, prevRotationDegrees,
                //         thisRotationDegreesBefore, thisRotationDegreesAfter);
                // }
                handRotations[i].set(thisQuaternion);
            }
        }
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

    private toCloneableLandmarks(
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

    private filterFaceLandmarks() {
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

    private lRLinkWeights() {
        const faceCameraAngle = degreeBetweenVectors(
            normalizedLandmarkToVector(this.faceNormal),
            new Vector3(0, 0, -1),
            true);
        const weightLeft = remapRangeWithCap(
            faceCameraAngle.y,
            Poses.LR_FACE_DIRECTION_RANGE,
            -Poses.LR_FACE_DIRECTION_RANGE,
            0, 1
        );
        const weightRight = remapRangeWithCap(
            faceCameraAngle.y,
            -Poses.LR_FACE_DIRECTION_RANGE,
            Poses.LR_FACE_DIRECTION_RANGE,
            0, 1
        );
        return {weightLeft, weightRight};
    }

    private lRLink(l: number, r: number) {
        const {weightLeft, weightRight} = this.lRLinkWeights();
        return weightLeft * l + weightRight * r;
    }

    private lRLinkVector(l: Vector3, r: Vector3) {
        const {weightLeft, weightRight} = this.lRLinkWeights();
        return l.scale(weightLeft).addInPlace(r.scale(weightRight));
    }

    private lRLinkQuaternion(l: Quaternion, r: Quaternion) {
        const {weightLeft, weightRight} = this.lRLinkWeights();
        return l.scale(weightLeft).addInPlace(r.scale(weightRight));
    }

    private static normalFromVertices(vertices: FilteredVectorLandmark3, reverse: boolean): Vector3 {
        if (reverse)
            vertices.reverse();
        const vec = [];
        for (let i = 0; i < 2; ++i) {
            vec.push(vertices[i + 1].pos.subtract(vertices[i].pos));
        }
        return vec[0].cross(vec[1]).normalize();
    }
}

enum LandmarkTimeIncrementMode {
    Universal,
    RealTime,
}

export class FilteredVectorLandmark {
    private oneEuroVectorFilter: OneEuroVectorFilter;
    private gaussianVectorFilter: Nullable<GaussianVectorFilter> = null;
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
        gaussianSigma?: number,
    ) {
        this.oneEuroVectorFilter = new OneEuroVectorFilter(
            this.t,
            this.pos,
            Vector3.Zero(),
            oneEuroCutoff,
            oneEuroBeta);
        if (gaussianSigma)
            this.gaussianVectorFilter = new GaussianVectorFilter(5, gaussianSigma);
        this.highPassFilter = new EuclideanHighPassFilter(highPassThreshold);
    }

    public updatePosition(dt: number, pos: Vector3, visibility?: number) {
        if (this._tIncrementMode === LandmarkTimeIncrementMode.Universal)
            this.t += 1;
        else if (this._tIncrementMode === LandmarkTimeIncrementMode.RealTime)
            this.t += dt;    // Assuming 60 fps

        // Face Mesh has no visibility
        if (!visibility || visibility > Poses.VISIBILITY_THRESHOLD) {
            pos = this.oneEuroVectorFilter.next(this.t, pos);

            if (this.gaussianVectorFilter) {
                this.gaussianVectorFilter.push(pos);
                pos = this.gaussianVectorFilter.apply();
            }

            this.highPassFilter.update(pos);
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
