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
    FACEMESH_LEFT_EYE,
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_LEFT_IRIS,
    FACEMESH_LIPS,
    FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW,
    FACEMESH_RIGHT_IRIS,
    NormalizedLandmark,
    NormalizedLandmarkList,
    POSE_LANDMARKS, POSE_LANDMARKS_LEFT, POSE_LANDMARKS_RIGHT,
    Results
} from "@mediapipe/holistic";
import {Nullable, Plane, Quaternion, Vector3} from "@babylonjs/core";
import {
    AXIS,
    Basis,
    calcAvgPlane,
    calcSphericalCoord,
    calcSphericalCoord0,
    cloneableQuaternionToQuaternion,
    degreeBetweenVectors,
    depthFirstSearch,
    EuclideanHighPassFilter,
    FACE_LANDMARK_LENGTH,
    GaussianVectorFilter,
    getBasis,
    HAND_LANDMARK_LENGTH,
    HAND_LANDMARKS, HAND_LANDMARKS_BONE_MAPPING,
    handLandMarkToBoneName,
    initArray,
    KeysMatching,
    normalizedLandmarkToVector,
    OneEuroVectorFilter,
    POSE_LANDMARK_LENGTH, projectVectorOnPlane,
    quaternionBetweenBases,
    quaternionToDegrees,
    ReadonlyKeys,
    remapRangeWithCap,
    removeRotationAxisWithCap,
    reverseRotation,
    sphericalToQuaternion,
    vectorToNormalizedLandmark
} from "../helper/utils";
import {TransformNodeTreeNode} from "v3d-core/dist/src/importer/babylon-vrm-loader/src";

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

    private readonly _baseBasis: Basis;
    get baseBasis(): Basis {
        return this._baseBasis;
    }

    constructor(
        q: Nullable<Quaternion>,
        basis?: Basis
    ) {
        if (q) this.set(q);
        this._baseBasis = basis ? basis : new Basis(null);
    }

    public set(q: Quaternion) {
        this.x = q.x;
        this.y = q.y;
        this.z = q.z;
        this.w = q.w;
    }

    public rotateBasis(q: Quaternion): Basis {
        return this._baseBasis.rotateByQuaternion(q);
    }
}
export interface CloneableQuaternionMap {
    [key: string]: CloneableQuaternion
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

export type PosesKey = keyof Omit<Poses, KeysMatching<Poses, Function> | ReadonlyKeys<Poses>>;

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

    /* Remap offsets to quaternions using arbitrary range.
     * IRIS_MP=MediaPipe Iris
     * IRIS_BJS=BabylonJS RotationYawPitchRoll
     */
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
    private bonesHierarchyTree: Nullable<TransformNodeTreeNode> = null;

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
                0.01, 0.6, 0.007);
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
                0.01, 15, 0.002);
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
            0.01, 2, 0.008);
    public inputLeftHandLandmarks: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        HAND_LANDMARK_LENGTH, () => {
            return {x: 0, y:0, z: 0};
        });
    private leftHandLandmarks: FilteredVectorLandmarkList = initArray<FilteredVectorLandmark>(
        HAND_LANDMARK_LENGTH, () => {
            return new FilteredVectorLandmark(
                0.001, 0.6,
                Poses.HAND_HIGH_PASS_THRESHOLD);
        });
    public cloneableLeftHandLandmarks: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        HAND_LANDMARK_LENGTH, () => {
            return {x: 0, y:0, z: 0};
        });

    // Right Hand Landmarks
    private rightWristOffset: FilteredVectorLandmark =
        new FilteredVectorLandmark(
            0.01, 2, 0.008);
    public inputRightHandLandmarks: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        HAND_LANDMARK_LENGTH, () => {
            return {x: 0, y:0, z: 0};
        });
    private rightHandLandmarks: FilteredVectorLandmarkList = initArray<FilteredVectorLandmark>(
        HAND_LANDMARK_LENGTH, () => {
            return new FilteredVectorLandmark(
                0.001, 0.6,
                Poses.HAND_HIGH_PASS_THRESHOLD);
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
    // TODO: option: lock x rotation
    private _irisQuaternion: CloneableQuaternionList = [
        new CloneableQuaternion(Quaternion.Identity()),
        new CloneableQuaternion(Quaternion.Identity()),
        new CloneableQuaternion(Quaternion.Identity()),
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
    private _boneRotations: CloneableQuaternionMap = {};
    get boneRotations(): CloneableQuaternionMap {
        return this._boneRotations;
    }
    private _leftHandNormals: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        3, () => {
            return {x: 0, y: 0, z: 0};
        });
    get leftHandNormals(): NormalizedLandmarkList {
        return this._leftHandNormals;
    }
    private _rightHandNormals: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        3, () => {
            return {x: 0, y: 0, z: 0};
        });
    get rightHandNormals(): NormalizedLandmarkList {
        return this._rightHandNormals;
    }
    private _poseNormals: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        3, // Arbitrary length for debugging
        () => {
            return {x: 0, y: 0, z: 0};
        });
    get poseNormals(): NormalizedLandmarkList {
        return this._poseNormals;
    }


    private midHipBase: Nullable<Vector3> = null;

    constructor() {}

    public setBonesHierarchyTree(tree: TransformNodeTreeNode) {
        // NOTE: always assumes bones have unique names
        if (this.bonesHierarchyTree) return;

        this.bonesHierarchyTree = tree;
        depthFirstSearch(this.bonesHierarchyTree, (n: TransformNodeTreeNode) => {
            this._boneRotations[n.name] = new CloneableQuaternion(
                Quaternion.Identity());
            return false;
        });
        this.initBoneRotations();
    }

    /*
     * All MediaPipe inputs have the following conventions:
     *  - Left-right mirrored
     *  - Face towards -Z (towards camera) by default
     */
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

        // Bone Orientations Independent
        // Calculate iris orientations
        this.calcIrisNormal();

        // Calculate expressions
        this.calcExpressions();

        // Bone Orientations Dependent
        // Calculate face orientation
        this.calcFaceNormal();

        // Calculate full body bones
        this.calcPoseBones();

        // Calculate hand bones
        this.calcHandBones();

        // Post processing
    }

    private getKeyPoints() {
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

        // Remap offsets to quaternions
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

    private calcPoseBones() {
        // Use hips as the starting point. Rotation of hips is always on XZ plane.
        // Neck and chest are derived from angle between spine and head.
        // Upper chest is not used.

        const leftHip = this.poseLandmarks[POSE_LANDMARKS.LEFT_HIP].pos;
        const rightHip = this.poseLandmarks[POSE_LANDMARKS.RIGHT_HIP].pos;
        const leftShoulder = this.poseLandmarks[POSE_LANDMARKS.LEFT_SHOULDER].pos;
        const rightShoulder = this.poseLandmarks[POSE_LANDMARKS.RIGHT_SHOULDER].pos;

        this.poseNormals.length = 0;

        // Hips normal
        const hipNormR = Plane.FromPoints(leftHip, rightHip, rightShoulder).normal;
        const hipNormL = Plane.FromPoints(leftHip, rightHip, rightShoulder).normal;
        const hipNormal = hipNormL.add(hipNormR).normalize();
        hipNormal.y = 0;
        this.poseNormals.push(vectorToNormalizedLandmark(hipNormal));

        // Chest/Shoulder normal
        const shoulderNormR = Plane.FromPoints(rightShoulder, leftShoulder, rightHip).normal;
        const shoulderNormL = Plane.FromPoints(rightShoulder, leftShoulder, leftHip).normal;
        const shoulderNormal = shoulderNormL.add(shoulderNormR).normalize();
        this.poseNormals.push(vectorToNormalizedLandmark(shoulderNormal));
        this.poseNormals.push(vectorToNormalizedLandmark(shoulderNormal));

        // Hips
        let [theta, phi] = calcSphericalCoord0(
            hipNormal, this._boneRotations['hips'].baseBasis);
        this._boneRotations['hips'].set(reverseRotation(sphericalToQuaternion(
            this._boneRotations['hips'].baseBasis, theta, phi), AXIS.y));
        const hipsRotationDegrees = quaternionToDegrees(cloneableQuaternionToQuaternion(
            this._boneRotations['hips']));

        // Spine
        const spineBasis = this._boneRotations['spine'].rotateBasis(
            this.applyQuaternionChain('spine', false));
        [theta, phi] = calcSphericalCoord0(
            shoulderNormal, spineBasis);
        this._boneRotations['spine'].set(reverseRotation(sphericalToQuaternion(
            this._boneRotations['spine'].baseBasis, theta, phi), AXIS.y));

        const lr = ["left", "right"];
        // Arms
        for (const k of lr) {
            const upperArmKey = `${k}UpperArm`;
            const lowerArmKey = `${k}LowerArm`;
            const shoulderLandmark = this.poseLandmarks[POSE_LANDMARKS[`${k.toUpperCase()}_SHOULDER` as keyof typeof POSE_LANDMARKS]].pos;
            const elbowLandmark = this.poseLandmarks[POSE_LANDMARKS[`${k.toUpperCase()}_ELBOW` as keyof typeof POSE_LANDMARKS]].pos;
            const wristLandmark = this.poseLandmarks[POSE_LANDMARKS[`${k.toUpperCase()}_WRIST` as keyof typeof POSE_LANDMARKS]].pos;

            const upperArmDir = elbowLandmark.subtract(shoulderLandmark).normalize();
            const upperArmBasis = this._boneRotations[upperArmKey].rotateBasis(
                this.applyQuaternionChain(upperArmKey, false));
            [theta, phi] = calcSphericalCoord0(upperArmDir, upperArmBasis);
            this._boneRotations[upperArmKey].set(reverseRotation(sphericalToQuaternion(
                this._boneRotations[upperArmKey].baseBasis, theta, phi), AXIS.yz));

            const lowerArmDir = wristLandmark.subtract(elbowLandmark).normalize();
            const lowerArmBasis = this._boneRotations[lowerArmKey].rotateBasis(
                this.applyQuaternionChain(lowerArmKey, false));
            [theta, phi] = calcSphericalCoord0(lowerArmDir, lowerArmBasis);
            this._boneRotations[lowerArmKey].set(reverseRotation(sphericalToQuaternion(
                this._boneRotations[lowerArmKey].baseBasis, theta, phi), AXIS.yz));
        }
        // Legs
        for (const k of lr) {
            const thisLandmarks = k === "left" ? POSE_LANDMARKS_LEFT : POSE_LANDMARKS_RIGHT;
            const upperLegKey = `${k}UpperLeg`;
            const lowerLegKey = `${k}LowerLeg`;
            const hipLandmark = this.poseLandmarks[thisLandmarks[`${k.toUpperCase()}_HIP` as keyof typeof thisLandmarks]].pos;
            const kneeLandmark = this.poseLandmarks[thisLandmarks[`${k.toUpperCase()}_KNEE` as keyof typeof thisLandmarks]].pos;
            const ankleLandmark = this.poseLandmarks[thisLandmarks[`${k.toUpperCase()}_ANKLE` as keyof typeof thisLandmarks]].pos;

            const upperLegDir = kneeLandmark.subtract(hipLandmark).normalize();
            const upperLegBasis = this._boneRotations[upperLegKey].rotateBasis(
                this.applyQuaternionChain(upperLegKey, false));
            [theta, phi] = calcSphericalCoord0(upperLegDir, upperLegBasis);
            this._boneRotations[upperLegKey].set(reverseRotation(sphericalToQuaternion(
                this._boneRotations[upperLegKey].baseBasis, theta, phi), AXIS.yz));

            const lowerLegDir = ankleLandmark.subtract(kneeLandmark).normalize();
            const lowerLegBasis = this._boneRotations[lowerLegKey].rotateBasis(
                this.applyQuaternionChain(lowerLegKey, false));
            [theta, phi] = calcSphericalCoord0(lowerLegDir, lowerLegBasis);
            this._boneRotations[lowerLegKey].set(reverseRotation(sphericalToQuaternion(
                this._boneRotations[lowerLegKey].baseBasis, theta, phi), AXIS.yz));
        }
    }

    private calcHandBones() {
        // Right hand shall have local x reversed?
        const hands = {
            left: this.leftHandLandmarks,
            right: this.rightHandLandmarks,
        }

        for (const [k, v] of Object.entries(hands)) {
            const isLeft = k === 'left';
            const vertices: FilteredVectorLandmark3[] = [
                [v[HAND_LANDMARKS.WRIST], v[HAND_LANDMARKS.PINKY_MCP], v[HAND_LANDMARKS.INDEX_FINGER_MCP]],
                [v[HAND_LANDMARKS.WRIST], v[HAND_LANDMARKS.RING_FINGER_MCP], v[HAND_LANDMARKS.INDEX_FINGER_MCP]],
                [v[HAND_LANDMARKS.WRIST], v[HAND_LANDMARKS.PINKY_MCP], v[HAND_LANDMARKS.MIDDLE_FINGER_MCP]],
            ]

            // Root normal
            const handNormalsKey = `${k}HandNormals`;
            const handNormals = this[handNormalsKey as PosesKey] as NormalizedLandmarkList;
            handNormals.length = 0;
            const rootNormal = vertices.reduce((prev, curr) => {
                const _normal = Poses.normalFromVertices(curr, isLeft);
                // handNormals.push(vectorToNormalizedLandmark(_normal));
                return prev.add(_normal);
            }, Vector3.Zero()).normalize();
            // handNormals.push(vectorToNormalizedLandmark(rootNormal));

            const thisWristRotation = this._boneRotations[handLandMarkToBoneName(HAND_LANDMARKS.WRIST, isLeft)];
            const basis1: Basis = thisWristRotation.baseBasis;

            // Project palm landmarks to average plane
            const projectedLandmarks = calcAvgPlane([
                v[HAND_LANDMARKS.WRIST].pos,
                v[HAND_LANDMARKS.INDEX_FINGER_MCP].pos,
                v[HAND_LANDMARKS.MIDDLE_FINGER_MCP].pos,
                v[HAND_LANDMARKS.RING_FINGER_MCP].pos,
                v[HAND_LANDMARKS.PINKY_MCP].pos
            ], rootNormal);
            const basis2 = getBasis([
                projectedLandmarks[0],
                projectedLandmarks[2],
                projectedLandmarks[4]
            ]).rotateByQuaternion(this.applyQuaternionChain(HAND_LANDMARKS.WRIST, isLeft).conjugate());
            const wristRotationQuaternionRaw = quaternionBetweenBases(basis1, basis2);
            // TODO: Debug
            // handNormals.push(vectorToNormalizedLandmark(axes1[0]));
            // handNormals.push(vectorToNormalizedLandmark(axes1[1]));
            // handNormals.push(vectorToNormalizedLandmark(axes1[2]));
            // handNormals.push(vectorToNormalizedLandmark(axes2[0]));
            // handNormals.push(vectorToNormalizedLandmark(axes2[1]));
            // handNormals.push(vectorToNormalizedLandmark(axes2[2]));

            const wristRotationQuaternion = reverseRotation(wristRotationQuaternionRaw, AXIS.yz);
            thisWristRotation.set(wristRotationQuaternion);
            // TODO: z rotation on arms
            const wristRotationDegrees = quaternionToDegrees(wristRotationQuaternion);

            for (let i = 1; i < HAND_LANDMARK_LENGTH; ++i) {
                if (i % 4 === 0) continue;

                const thisHandRotation = this._boneRotations[handLandMarkToBoneName(i, isLeft)];
                const thisLandmark = v[i].pos.clone();
                const nextLandmark = v[i + 1].pos.clone();
                let thisDir = nextLandmark.subtract(thisLandmark).normalize();

                // if (i === 10) {
                //     const prevQuaternion1 = Quaternion.Identity();
                //
                //     // 16
                //     prevQuaternion1.multiplyInPlace(cloneableQuaternionToQuaternion(
                //         this._boneRotations[handLandMarkToBoneName(HAND_LANDMARKS.WRIST, isLeft)]));
                //     prevQuaternion1.normalize();
                //     const newBasis1 = this._boneRotations[handLandMarkToBoneName(i, isLeft)]
                //         .rotateBasis(prevQuaternion1);
                //
                //     // 17
                //     const prevQuaternion2 = Quaternion.Identity();
                //     prevQuaternion2.multiplyInPlace(reverseRotation(
                //         cloneableQuaternionToQuaternion(
                //             this._boneRotations[handLandMarkToBoneName(HAND_LANDMARKS.WRIST, isLeft)]),
                //         AXIS.yz));
                //     prevQuaternion2.multiplyInPlace(reverseRotation(
                //         cloneableQuaternionToQuaternion(
                //             this._boneRotations[handLandMarkToBoneName(9, isLeft)]),
                //         AXIS.yz));
                //     // prevQuaternion2.multiplyInPlace(reverseRotation(
                //     //     cloneableQuaternionToQuaternion(handRotations[2]), AXIS.yz));
                //     prevQuaternion2.normalize();
                //     const newBasis2 = this._boneRotations[handLandMarkToBoneName(i, isLeft)]
                //         .rotateBasis(prevQuaternion2);
                //
                //     handNormals.push(vectorToNormalizedLandmark(newBasis1.x));
                //     handNormals.push(vectorToNormalizedLandmark(newBasis1.y));
                //     handNormals.push(vectorToNormalizedLandmark(newBasis1.z));
                //
                //     const qToOriginal1 = Quaternion.Inverse(Quaternion.FromRotationMatrix(
                //         newBasis1.asMatrix())).normalize();
                //     const posInOriginal1 = Vector3.Zero();
                //     thisDir.rotateByQuaternionToRef(qToOriginal1, posInOriginal1);
                //     posInOriginal1.normalize();
                //     handNormals.push(vectorToNormalizedLandmark(thisDir));
                //     handNormals.push(vectorToNormalizedLandmark(posInOriginal1));
                //
                //     handNormals.push(vectorToNormalizedLandmark(newBasis2.x));
                //     handNormals.push(vectorToNormalizedLandmark(newBasis2.y));
                //     handNormals.push(vectorToNormalizedLandmark(newBasis2.z));
                //
                //     const qToOriginal2 = Quaternion.Inverse(Quaternion.FromRotationMatrix(
                //         newBasis2.asMatrix())).normalize();
                //     const posInOriginal2 = Vector3.Zero();
                //     thisDir.rotateByQuaternionToRef(qToOriginal2, posInOriginal2);
                //     posInOriginal2.normalize();
                //     handNormals.push(vectorToNormalizedLandmark(thisDir));
                //     handNormals.push(vectorToNormalizedLandmark(posInOriginal2));
                // }

                const prevQuaternion = this.applyQuaternionChain(i, isLeft);
                const thisBasis = thisHandRotation.rotateBasis(prevQuaternion);

                // Project landmark to XZ plane for second and third segments
                if (i % 4 === 2 || i % 4 === 3) {
                    const projPlane = Plane.FromPositionAndNormal(
                        Vector3.Zero(), thisBasis.y.clone());
                    thisDir = projectVectorOnPlane(projPlane, thisDir);
                }
                let [theta, phi] = calcSphericalCoord(thisDir, thisBasis);

                // Need to use original Basis, because the quaternion from
                // RotationAxis inherently uses local coordinate system.
                let thisRotationQuaternion;
                    const lrCoeff = isLeft ? -1 : 1;
                    // Thumb rotations are y main. Others are z main.
                const removeAxis = i % 4 === 1 ?
                    i < 4 ? AXIS.none : AXIS.x
                    :
                    i < 4 ? AXIS.xz : AXIS.xy;
                const firstCapAxis = i < 4 ?
                    AXIS.z : AXIS.y;
                const secondCapAxis = i < 4 ?
                    AXIS.y : AXIS.z;
                thisRotationQuaternion =
                    removeRotationAxisWithCap(
                        sphericalToQuaternion(thisHandRotation.baseBasis, theta, phi),
                        removeAxis,
                        firstCapAxis, -15, 15,
                        secondCapAxis, lrCoeff * -15, lrCoeff * 110);
                thisRotationQuaternion = reverseRotation(thisRotationQuaternion, AXIS.yz);
                const thisRotationQuaternionDegrees = quaternionToDegrees(thisRotationQuaternion);
                thisHandRotation.set(thisRotationQuaternion);
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
            // TODO: positional movement
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

        // TODO: update wrist offset only when debugging
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

    private initHandBoneRotations(isLeft: boolean) {
        // TODO: adjust bases
        // Wrist's basis is used for calculating quaternion between two Cartesian coordinate systems directly
        // All others' are used for rotating planes of a Spherical coordinate system at the node
        this._boneRotations[handLandMarkToBoneName(HAND_LANDMARKS.WRIST, isLeft)] =
            new CloneableQuaternion(
                Quaternion.Identity(), isLeft ? getBasis(
                    [
                        new Vector3(0, 0, 0),
                        new Vector3(isLeft ? 1 : -1, 0, 0),
                        new Vector3(isLeft ? 1 : -1, 0, 1)
                    ]) : new Basis([
                    new Vector3(-0.9327159079568041, 0.12282522615654383, -0.3390501421086685).normalize(),
                    new Vector3(-0.010002212677077182, 0.0024727643453822945, 0.028411551927747327).normalize(),
                    new Vector3(0.14320801411112857, 0.9890497926949048, -0.03566472016590984).normalize()
                ]));
        // Thumb
        // for (let i = 0; i < 4; ++i) {
        //     ret.push(new CloneableQuaternion(
        //         Quaternion.Identity(),
        //         // new Basis([
        //         //     new Vector3(isLeft ? 1 : -1, -1, -1),
        //         //     new Vector3(isLeft ? -1 : 1, 1, 1),
        //         //     new Vector3(isLeft ? -1 : 1, 1, -1),
        //         // ])
        //         getBasis(
        //             [
        //                 new Vector3(0, 0, 0),
        //                 new Vector3(isLeft ? 0.7 : -0.7, -0.1, -1),
        //                 new Vector3(-1, 0, isLeft ? -1 : 1)
        //             ])
        //     ));
        // }
        // THUMB_CMC
        // THUMB_MCP
        // THUMB_IP
        for (let i = 1; i < 4; ++i) {

            // const tMCP_X = new Vector3(isLeft ? 1 : -1, 0, -0.5);
            // const tMCP_Z = new Vector3(0, 1, 0);
            // const tMCP_Y = Vector3.Cross(tMCP_Z, tMCP_X);
            const tMCP_X = new Vector3(isLeft ? 1 : -1, 0, -1.5).normalize();
            const tMCP_Y = new Vector3(0, isLeft ? -1 : 1, 0);
            const tMCP_Z = Vector3.Cross(tMCP_X, tMCP_Y).normalize();
            this._boneRotations[handLandMarkToBoneName(i, isLeft)] =
                new CloneableQuaternion(
                    Quaternion.Identity(), new Basis([
                        tMCP_X,
                        // new Vector3(0, 0, isLeft ? -1 : 1),
                        tMCP_Y,
                        tMCP_Z,
                    ]));
        }
        // Index
        for (let i = 5; i < 8; ++i) {
            this._boneRotations[handLandMarkToBoneName(i, isLeft)] =
                new CloneableQuaternion(
                    Quaternion.Identity(), new Basis([
                        new Vector3(isLeft ? 1 : -1, 0, 0),
                        new Vector3(0, 0, isLeft ? -1 : 1),
                        new Vector3(0, 1, 0),
                    ]));
        }
        // Middle
        for (let i = 9; i < 12; ++i) {
            this._boneRotations[handLandMarkToBoneName(i, isLeft)] =
                new CloneableQuaternion(
                    Quaternion.Identity(), new Basis([
                        new Vector3(isLeft ? 1 : -1, 0, 0),
                        new Vector3(0, 0, isLeft ? -1 : 1),
                        new Vector3(0, 1, 0),
                    ]));
        }
        // Ring
        for (let i = 13; i < 16; ++i) {
            this._boneRotations[handLandMarkToBoneName(i, isLeft)] =
                new CloneableQuaternion(
                    Quaternion.Identity(), new Basis([
                        new Vector3(isLeft ? 1 : -1, 0, 0),
                        new Vector3(0, 0, isLeft ? -1 : 1),
                        new Vector3(0, 1, 0),
                    ]));
        }
        // Pinky
        for (let i = 17; i < 20; ++i) {
            this._boneRotations[handLandMarkToBoneName(i, isLeft)] =
                new CloneableQuaternion(
                    Quaternion.Identity(), new Basis([
                        new Vector3(isLeft ? 1 : -1, 0, 0),
                        new Vector3(0, 0, isLeft ? -1 : 1),
                        new Vector3(0, 1, 0),
                    ]));
        }
    }

    private initBoneRotations() {
        // Hand bones
        this.initHandBoneRotations(true);
        this.initHandBoneRotations(false);

        // Pose bones
        this._boneRotations['hips'] = new CloneableQuaternion(
            Quaternion.Identity(), new Basis([
                new Vector3(0, 0, -1),
                new Vector3(-1, 0, 0),
                new Vector3(0, 1, 0),
            ])
        );
        this._boneRotations['spine'] = new CloneableQuaternion(
            Quaternion.Identity(), new Basis([
                new Vector3(0, 0, -1),
                new Vector3(-1, 0, 0),
                new Vector3(0, 1, 0),
            ])
        );

        const lr = ["left", "right"];
        // Arms
        for (const k of lr) {
            const isLeft = k === "left";
            this._boneRotations[`${k}UpperArm`] = new CloneableQuaternion(
                Quaternion.Identity(), new Basis([
                    new Vector3(isLeft ? 1 : -1, 0, 0),
                    new Vector3(0, 0, isLeft ? -1 : 1),
                    new Vector3(0, 1, 0),
                ]));
            this._boneRotations[`${k}LowerArm`] = new CloneableQuaternion(
                Quaternion.Identity(), new Basis([
                    new Vector3(isLeft ? 1 : -1, 0, 0),
                    new Vector3(0, 0, isLeft ? -1 : 1),
                    new Vector3(0, 1, 0),
                ]));
        }
        // Legs
        // TODO: make basis towards inside a little
        for (const k of lr) {
            this._boneRotations[`${k}UpperLeg`] = new CloneableQuaternion(
                Quaternion.Identity(), new Basis([
                    new Vector3(0, -1, 0),
                    new Vector3(-1, 0, 0),
                    new Vector3(0, 0, -1),
                ]));
            this._boneRotations[`${k}LowerLeg`] = new CloneableQuaternion(
                Quaternion.Identity(), new Basis([
                    new Vector3(0, -1, 0),
                    new Vector3(-1, 0, 0),
                    new Vector3(0, 0, -1),
                ]));
        }
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

    // Recursively apply previous quaternions to current basis
    private applyQuaternionChain(startLandmark: number | string, isLeft: boolean): Quaternion {
        const q = Quaternion.Identity();
        const rotations: Quaternion[] = [];
        let [startNode, parentMap]: [
            TransformNodeTreeNode, Map<TransformNodeTreeNode, TransformNodeTreeNode>
        ] = depthFirstSearch(this.bonesHierarchyTree, (n: TransformNodeTreeNode) => {
            const targetName = Number.isFinite(startLandmark) ?
                handLandMarkToBoneName(startLandmark as number, isLeft)
                : startLandmark;
            return (n.name === targetName);
        });
        while (parentMap.has(startNode)) {
            startNode = parentMap.get(startNode)!;
            const mirrorYZ = startNode.name !== "spine" && startNode.name !== "hips";
            const boneQuaternion = this._boneRotations[startNode.name];
            rotations.push(reverseRotation(
                cloneableQuaternionToQuaternion(boneQuaternion),
                mirrorYZ ? AXIS.yz : AXIS.y));
        }
        // Quaternions need to be applied from parent to children
        rotations.reverse().map((tq: Quaternion) => {
            q.multiplyInPlace(tq);
        });
        // q.multiplyInPlace(reverseRotation(
        //     cloneableQuaternionToQuaternion(
        //         this._boneRotations[handLandMarkToBoneName(HAND_LANDMARKS.WRIST, isLeft)]),
        //     AXIS.yz));
        // const iStart = Math.floor(i / 4) * 4 + 1;
        // for (let idx = iStart; idx < i; ++idx) {
        //     q.multiplyInPlace(reverseRotation(
        //         cloneableQuaternionToQuaternion(
        //             this._boneRotations[handLandMarkToBoneName(idx, isLeft)]),
        //         i < 4 ? AXIS.none : AXIS.yz));
        // }
        // const q = Quaternion.FromRotationMatrix(
        //     thisHandBoneWorldMatrices[(i - 1) % 4 === 0 ? 0 : i - 1]
        //         .getRotationMatrix())
        q.normalize();

        return q;
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
        if (visibility === undefined || visibility > Poses.VISIBILITY_THRESHOLD) {
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
