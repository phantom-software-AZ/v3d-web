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
    FACEMESH_LEFT_EYE,
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_LEFT_IRIS,
    FACEMESH_LIPS,
    FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW,
    FACEMESH_RIGHT_IRIS,
    NormalizedLandmark,
    NormalizedLandmarkList,
    POSE_LANDMARKS,
    POSE_LANDMARKS_LEFT,
    POSE_LANDMARKS_RIGHT,
    Results
} from "@mediapipe/holistic";
import {Curve3, Nullable, Plane, Quaternion, Vector3} from "@babylonjs/core";
import {
    initArray,
    KeysMatching,
    projectVectorOnPlane,
    ReadonlyKeys,
    remapRangeWithCap,
} from "../helper/utils";
import {TransformNodeTreeNode} from "v3d-core/dist/src/importer/babylon-vrm-loader/src";
import {
    CloneableResults, depthFirstSearch,
    FACE_LANDMARK_LENGTH, FilteredVectorLandmark, FilteredVectorLandmark3, FilteredVectorLandmarkList,
    HAND_LANDMARK_LENGTH, HAND_LANDMARKS, handLandMarkToBoneName, normalizedLandmarkToVector,
    POSE_LANDMARK_LENGTH,
    vectorToNormalizedLandmark
} from "../helper/landmark";
import {
    AXIS,
    calcSphericalCoord,
    calcSphericalCoord0,
    CloneableQuaternion,
    CloneableQuaternionList,
    CloneableQuaternionMap,
    cloneableQuaternionToQuaternion, degreeBetweenVectors,
    quaternionToDegrees,
    removeRotationAxisWithCap,
    reverseRotation,
    sphericalToQuaternion
} from "../helper/quaternion";
import {Basis, calcAvgPlane, getBasis, quaternionBetweenBases} from "../helper/basis";
import {VISIBILITY_THRESHOLD} from "../helper/filter";


export class PoseKeyPoints {
    public top_face_oval = new FilteredVectorLandmark();
    public left_face_oval = new FilteredVectorLandmark();
    public bottom_face_oval = new FilteredVectorLandmark();
    public right_face_oval = new FilteredVectorLandmark();
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
    public static readonly FACE_MESH_CONNECTIONS = [
        FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYEBROW,
        FACEMESH_LEFT_EYE, FACEMESH_RIGHT_EYE,
        FACEMESH_LEFT_IRIS, FACEMESH_RIGHT_IRIS,
        FACEMESH_LIPS, FACEMESH_FACE_OVAL
    ];
    public static readonly HAND_BASE_ROOT_NORMAL = new Vector3(0, -1, 0);


    private static readonly HAND_POSITION_SCALING = 0.8;

    /* Remap offsets to quaternions using arbitrary range.
     * IRIS_MP=MediaPipe Iris
     * IRIS_BJS=BabylonJS RotationYawPitchRoll
     */
    private static readonly IRIS_MP_X_RANGE = 0.027;
    private static readonly IRIS_MP_Y_RANGE = 0.011;
    private static readonly IRIS_BJS_X_RANGE = 0.28;
    private static readonly IRIS_BJS_Y_RANGE = 0.22;

    private static readonly BLINK_EYE_WIDTH_CURVE_LOW = Curve3.CreateCatmullRomSpline([
        new Vector3(0.105, 0.0189, 0),
        new Vector3(0.058, 0.018, 0),
        new Vector3(0.016,  0.0144, 0),
        new Vector3(0.0105, 0.0165, 0),
    ], 100);
    private static readonly BLINK_EYE_WIDTH_CURVE_HIGH = Curve3.CreateCatmullRomSpline([
        new Vector3(0.105, 0.0215, 0),
        new Vector3(0.058, 0.0198, 0),
        new Vector3(0.016,  0.015, 0),
        new Vector3(0.0105, 0.0205, 0),
    ], 100);
    private static readonly MOUTH_MP_RANGE_LOW = 0.001;
    private static readonly MOUTH_MP_RANGE_HIGH = 0.06;

    private static readonly EYE_WIDTH_BASELINE = 0.0546;
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
            return new FilteredVectorLandmark({
                R: 1, Q: 1, type: 'Kalman',
            });
        });
    private worldPoseLandmarks: FilteredVectorLandmarkList = initArray<FilteredVectorLandmark>(
        POSE_LANDMARK_LENGTH, () => {
            return new FilteredVectorLandmark({
                R: 1, Q: 1, type: 'Kalman',
            });  // 0.01, 0.6, 0.007
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
            return new FilteredVectorLandmark({
                // oneEuroCutoff: 0.01, oneEuroBeta: 15, type: 'OneEuro',
                R: 1, Q: 10, type: 'Kalman',
                gaussianSigma: 2
            });     // 0.01, 15, 0.002
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
        new FilteredVectorLandmark({
            R: 0.1, Q: 2, type: 'Kalman',
        });    // 0.01, 2, 0.008
    public inputLeftHandLandmarks: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        HAND_LANDMARK_LENGTH, () => {
            return {x: 0, y:0, z: 0};
        });
    private leftHandLandmarks: FilteredVectorLandmarkList = initArray<FilteredVectorLandmark>(
        HAND_LANDMARK_LENGTH, () => {
            return new FilteredVectorLandmark({
                R: 1, Q: 10, type: 'Kalman',
            });    // 0.001, 0.6

        });
    public cloneableLeftHandLandmarks: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        HAND_LANDMARK_LENGTH, () => {
            return {x: 0, y:0, z: 0};
        });
    private leftHandNormal: Vector3 = Vector3.Zero();

    // Right Hand Landmarks
    private rightWristOffset: FilteredVectorLandmark =
        new FilteredVectorLandmark({
            R: 0.1, Q: 2, type: 'Kalman',
        });    // 0.01, 2, 0.008
    public inputRightHandLandmarks: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        HAND_LANDMARK_LENGTH, () => {
            return {x: 0, y:0, z: 0};
        });
    private rightHandLandmarks: FilteredVectorLandmarkList = initArray<FilteredVectorLandmark>(
        HAND_LANDMARK_LENGTH, () => {
            return new FilteredVectorLandmark({
                R: 1, Q: 10, type: 'Kalman',
            });    // 0.001, 0.6
        });
    public cloneableRightHandLandmarks: NormalizedLandmarkList = initArray<NormalizedLandmark>(
        HAND_LANDMARK_LENGTH, () => {
            return {x: 0, y:0, z: 0};
        });
    private rightHandNormal: Vector3 = Vector3.Zero();

    // Feet
    private leftFootNormal: Vector3 = Vector3.Zero();
    private rightFootNormal: Vector3 = Vector3.Zero();
    private leftFootBasisRotation: Quaternion = Quaternion.Identity();
    private rightFootBasisRotation: Quaternion = Quaternion.Identity();

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

    public midHipOffset: NormalizedLandmark = {x: 0, y: 0, z: 0};

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

    /**
     * All MediaPipe inputs have the following conventions:
     *  - Left-right mirrored
     *  - Face towards -Z (towards camera) by default
     */
    public process(results: CloneableResults) {
        this.cloneableInputResults = results;
        if (!this.cloneableInputResults) return;

        this.preProcessResults();

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

        // Bone Orientations Dependent
        // Calculate face orientation
        this.calcFaceBones();

        // Calculate expressions
        this.calcExpressions();

        // Calculate full body bones
        this.calcPoseBones();

        // Calculate hand bones
        this.calcHandBones();

        // Post processing
    }

    private getKeyPoints() {
        // Get points from face mesh
        this._keyPoints.top_face_oval = this.faceLandmarks[this.faceMeshLandmarkIndexList[7][0]];
        this._keyPoints.left_face_oval = this.faceLandmarks[this.faceMeshLandmarkIndexList[7][6]];
        this._keyPoints.bottom_face_oval = this.faceLandmarks[this.faceMeshLandmarkIndexList[7][18]];
        this._keyPoints.right_face_oval = this.faceLandmarks[this.faceMeshLandmarkIndexList[7][30]];

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
    private calcFaceBones() {
        const axisX = this._keyPoints.left_face_oval.pos.subtract(this._keyPoints.right_face_oval.pos).normalize();
        const axisY = this._keyPoints.top_face_oval.pos.subtract(this._keyPoints.bottom_face_oval.pos).normalize();
        if (axisX.length() === 0 || axisY.length() === 0) return;
        const thisBasis = new Basis([
            axisX, axisY, Vector3.Cross(axisX, axisY)
        ]);

        // Distribute rotation between neck and head
        const headBasis = this._boneRotations['head'].rotateBasis(
            this.applyQuaternionChain('neck', false));
        const quaternion = reverseRotation(quaternionBetweenBases(thisBasis, headBasis), AXIS.x);
        quaternion.scaleInPlace(0.6);
        this._boneRotations['head'].set(quaternion);
        this._boneRotations['neck'].set(quaternion);
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

    private findPoint(curve: Curve3, x: number, eps=0.001) {
        const pts = curve.getPoints();
        if (x > pts[0].x) return pts[0].y;
        else if (x < pts[pts.length - 1].x) return pts[pts.length - 1].y;
        for (let i = 0; i < pts.length; ++i) {
            if (Math.abs(x - pts[i].x) < eps) return pts[i].y;
        }
        return 0;
    }

    private calcExpressions() {
        if (!this.cloneableInputResults?.faceLandmarks) return;

        const leftEyeWidth = this._keyPoints.left_eye_inner.pos.subtract(this._keyPoints.left_eye_outer.pos).length();
        console.log(leftEyeWidth,
            this.findPoint(Poses.BLINK_EYE_WIDTH_CURVE_HIGH, leftEyeWidth),
            this._keyPoints.left_eye_top.pos
            .subtract(this._keyPoints.left_eye_bottom.pos)
            .length() * Poses.EYE_WIDTH_BASELINE / leftEyeWidth);
        this._blinkLeft = 1 - remapRangeWithCap(
            this._keyPoints.left_eye_top.pos
                .subtract(this._keyPoints.left_eye_bottom.pos)
                .length() * Poses.EYE_WIDTH_BASELINE / leftEyeWidth,
            this.findPoint(Poses.BLINK_EYE_WIDTH_CURVE_LOW, leftEyeWidth),
            this.findPoint(Poses.BLINK_EYE_WIDTH_CURVE_HIGH, leftEyeWidth),
            0, 1
        );
        const rightEyeWidth = this._keyPoints.right_eye_inner.pos.subtract(this._keyPoints.right_eye_outer.pos).length();
        this._blinkRight = 1 - remapRangeWithCap(
            this._keyPoints.right_eye_top.pos
                .subtract(this._keyPoints.right_eye_bottom.pos)
                .length() * Poses.EYE_WIDTH_BASELINE / rightEyeWidth,
            this.findPoint(Poses.BLINK_EYE_WIDTH_CURVE_LOW, rightEyeWidth),
            this.findPoint(Poses.BLINK_EYE_WIDTH_CURVE_HIGH, rightEyeWidth),
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
        if (!this.cloneableInputResults?.poseLandmarks) return;
        // Use hips as the starting point. Rotation of hips is always on XZ plane.
        // Neck and chest are derived from angle between spine and head.
        // Upper chest is not used.

        const leftHip = this.worldPoseLandmarks[POSE_LANDMARKS.LEFT_HIP].pos;
        const rightHip = this.worldPoseLandmarks[POSE_LANDMARKS.RIGHT_HIP].pos;
        const leftShoulder = this.worldPoseLandmarks[POSE_LANDMARKS.LEFT_SHOULDER].pos;
        const rightShoulder = this.worldPoseLandmarks[POSE_LANDMARKS.RIGHT_SHOULDER].pos;

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

        this.calcWristBones();

        const lr = ["left", "right"];
        // Arms
        for (const k of lr) {
            const isLeft = k === "left";
            if (!this.shallUpdateArm(isLeft)) continue;

            const upperArmKey = `${k}UpperArm`;
            const shoulderLandmark = this.worldPoseLandmarks[POSE_LANDMARKS[`${k.toUpperCase()}_SHOULDER` as keyof typeof POSE_LANDMARKS]].pos;
            const elbowLandmark = this.worldPoseLandmarks[POSE_LANDMARKS[`${k.toUpperCase()}_ELBOW` as keyof typeof POSE_LANDMARKS]].pos;
            const wristLandmark = this.worldPoseLandmarks[POSE_LANDMARKS[`${k.toUpperCase()}_WRIST` as keyof typeof POSE_LANDMARKS]].pos;

            const upperArmDir = elbowLandmark.subtract(shoulderLandmark).normalize();
            const upperArmBasis = this._boneRotations[upperArmKey].rotateBasis(
                this.applyQuaternionChain(upperArmKey, false));
            [theta, phi] = calcSphericalCoord0(upperArmDir, upperArmBasis);
            this._boneRotations[upperArmKey].set(reverseRotation(sphericalToQuaternion(
                this._boneRotations[upperArmKey].baseBasis, theta, phi), AXIS.yz));

            // Rotate lower arms around X axis together with hands.
            // This is a combination of spherical coordinates rotation and rotation between bases.
            const handNormal = isLeft ? this.leftHandNormal : this.rightHandNormal;
            const lowerArmKey = `${k}LowerArm`;
            const lowerArmDir = wristLandmark.subtract(elbowLandmark).normalize();
            const lowerArmPrevQuaternion = this.applyQuaternionChain(lowerArmKey, false);
            const lowerArmBasis = this._boneRotations[lowerArmKey].rotateBasis(
                lowerArmPrevQuaternion);
            [theta, phi] = calcSphericalCoord0(lowerArmDir, lowerArmBasis);
            const firstQuaternion = reverseRotation(sphericalToQuaternion(
                this._boneRotations[lowerArmKey].baseBasis, theta, phi), AXIS.yz);
            const finalQuaternion = this.applyXRotationWithChild(
                lowerArmKey, lowerArmPrevQuaternion, firstQuaternion,
                handNormal, lowerArmBasis);

            const handNormalsKey = `${k}HandNormals`;
            const handNormals = this[handNormalsKey as PosesKey] as NormalizedLandmarkList;
            handNormals.length = 0;

            this._boneRotations[lowerArmKey].set(finalQuaternion);
        }
        // Update rotations on wrists
        this.calcWristBones(false);

        // Legs and feet
        this.calcFeetBones();
        if (!this.shallUpdateLegs()) return;
        for (const k of lr) {
            const isLeft = k === "left";

            const thisLandmarks = isLeft ? POSE_LANDMARKS_LEFT : POSE_LANDMARKS_RIGHT;
            const upperLegKey = `${k}UpperLeg`;
            const lowerLegKey = `${k}LowerLeg`;
            const hipLandmark = this.worldPoseLandmarks[thisLandmarks[`${k.toUpperCase()}_HIP` as keyof typeof thisLandmarks]].pos;
            const kneeLandmark = this.worldPoseLandmarks[thisLandmarks[`${k.toUpperCase()}_KNEE` as keyof typeof thisLandmarks]].pos;
            const ankleLandmark = this.worldPoseLandmarks[thisLandmarks[`${k.toUpperCase()}_ANKLE` as keyof typeof thisLandmarks]].pos;

            const upperLegDir = kneeLandmark.subtract(hipLandmark).normalize();
            const upperLegBasis = this._boneRotations[upperLegKey].rotateBasis(
                this.applyQuaternionChain(upperLegKey, false));
            [theta, phi] = calcSphericalCoord0(upperLegDir, upperLegBasis);
            this._boneRotations[upperLegKey].set(reverseRotation(sphericalToQuaternion(
                this._boneRotations[upperLegKey].baseBasis, theta, phi), AXIS.yz));

            const footNormal = k === "left" ? this.leftFootNormal : this.rightFootNormal;
            const lowerLegDir = ankleLandmark.subtract(kneeLandmark).normalize();
            const lowerLegPrevQuaternion = this.applyQuaternionChain(lowerLegKey, false);
            const lowerLegBasis = this._boneRotations[lowerLegKey].rotateBasis(
                lowerLegPrevQuaternion);
            [theta, phi] = calcSphericalCoord0(lowerLegDir, lowerLegBasis);
            const firstQuaternion = reverseRotation(sphericalToQuaternion(
                this._boneRotations[lowerLegKey].baseBasis, theta, phi), AXIS.yz);
            const finalQuaternion = this.applyXRotationWithChild(
                lowerLegKey, lowerLegPrevQuaternion, firstQuaternion,
                footNormal, lowerLegBasis);
            this._boneRotations[lowerLegKey].set(finalQuaternion);
        }

        this.calcFeetBones();
    }

    /*
     * thisKey: key in _boneRotations
     * prevQuaternion: Cumulated rotation quaternion up to parent
     * firstQuaternion: Rotation quaternion calculated without applying X rotation
     * normal: A normal pointing to -y
     * thisBasis: basis on this node after prevQuaternion is applied
     */
    private applyXRotationWithChild(
        thisKey: string,
        prevQuaternion: Quaternion,
        firstQuaternion: Quaternion,
        normal: Vector3,
        thisBasis: Basis,
        coeff = 0.5
    ) {
        const thisRotatedBasis = this._boneRotations[thisKey].rotateBasis(
            prevQuaternion.multiply(reverseRotation(firstQuaternion, AXIS.yz)));

        const thisYZPlane = Plane.FromPositionAndNormal(Vector3.Zero(), thisRotatedBasis.x.clone());
        const projectedNormal = Vector3.Zero();
        projectVectorOnPlane(thisYZPlane, normal).rotateByQuaternionToRef(
            Quaternion.Inverse(Quaternion.RotationQuaternionFromAxis(
                thisRotatedBasis.x.clone(), thisRotatedBasis.y.clone(), thisRotatedBasis.z.clone()
            )), projectedNormal);
        const projectedPrevZ = Vector3.Zero();
        projectVectorOnPlane(thisYZPlane, thisBasis.z.negate()).rotateByQuaternionToRef(
            Quaternion.Inverse(Quaternion.RotationQuaternionFromAxis(
                thisRotatedBasis.x.clone(), thisRotatedBasis.y.clone(), thisRotatedBasis.z.clone()
            )), projectedPrevZ);
        projectedPrevZ.normalize();
        const xPrev = Math.atan2(projectedPrevZ.y, -projectedPrevZ.z);
        const xAngle = Math.atan2(projectedNormal.y, -projectedNormal.z);
        // const xAngle = Math.atan2(projectedNormal.y, -projectedNormal.z) + Math.PI / 4;
        const thisXRotatedBasis = thisRotatedBasis.rotateByQuaternion(
            Quaternion.RotationAxis(thisRotatedBasis.x.clone(), (xAngle - xPrev) * coeff));
        // The quaternion needs to be calculated in original coordinate system
        const secondQuaternion = quaternionBetweenBases(
            thisBasis, thisXRotatedBasis, prevQuaternion
        );

        const finalQuaternion = reverseRotation(secondQuaternion, AXIS.yz);
        return finalQuaternion;
    }

    private calcWristBones(firstPass = true) {
        const hands = {
            left: this.leftHandLandmarks,
            right: this.rightHandLandmarks,
        }

        for (const [k, v] of Object.entries(hands)) {
            const isLeft = k === 'left';
            const wristVisilibity = this.cloneableInputResults?.poseLandmarks[
                isLeft ? POSE_LANDMARKS.LEFT_WRIST : POSE_LANDMARKS.RIGHT_WRIST].visibility || 0;
            if (wristVisilibity <= VISIBILITY_THRESHOLD) continue;

            const vertices: FilteredVectorLandmark3[] = [
                [v[HAND_LANDMARKS.WRIST], v[HAND_LANDMARKS.PINKY_MCP], v[HAND_LANDMARKS.INDEX_FINGER_MCP]],
                [v[HAND_LANDMARKS.WRIST], v[HAND_LANDMARKS.RING_FINGER_MCP], v[HAND_LANDMARKS.INDEX_FINGER_MCP]],
                [v[HAND_LANDMARKS.WRIST], v[HAND_LANDMARKS.PINKY_MCP], v[HAND_LANDMARKS.MIDDLE_FINGER_MCP]],
            ]

            // Root normal
            const handNormal = isLeft ? this.leftHandNormal : this.rightHandNormal;
            const rootNormal = vertices.reduce((prev, curr) => {
                const _normal = Poses.normalFromVertices(curr, isLeft);
                // handNormals.push(vectorToNormalizedLandmark(_normal));
                return prev.add(_normal);
            }, Vector3.Zero()).normalize();
            handNormal.copyFrom(rootNormal);
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
                projectedLandmarks[1],
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
            const wristRotationDegrees = quaternionToDegrees(wristRotationQuaternion);
            if (!firstPass)
                thisWristRotation.set(wristRotationQuaternion);
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

            for (let i = 1; i < HAND_LANDMARK_LENGTH; ++i) {
                if (i % 4 === 0) continue;

                const thisHandRotation = this._boneRotations[handLandMarkToBoneName(i, isLeft)];
                const thisLandmark = v[i].pos.clone();
                const nextLandmark = v[i + 1].pos.clone();
                let thisDir = nextLandmark.subtract(thisLandmark).normalize();

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
                const firstCapAxis = i < 4 ? AXIS.z : AXIS.y;
                const secondCapAxis = i < 4 ? AXIS.y : AXIS.z;
                thisRotationQuaternion =
                    removeRotationAxisWithCap(
                        sphericalToQuaternion(thisHandRotation.baseBasis, theta, phi),
                        removeAxis,
                        firstCapAxis, -15, 15,
                        secondCapAxis, lrCoeff * -15, lrCoeff * 110);
                thisRotationQuaternion = reverseRotation(thisRotationQuaternion, AXIS.yz);
                thisHandRotation.set(thisRotationQuaternion);
            }
        }
    }

    private calcFeetBones() {
        if (!this.shallUpdateLegs()) return;
        const feet = {
            left: getBasis([
                this.worldPoseLandmarks[POSE_LANDMARKS_LEFT.LEFT_HEEL].pos,
                this.worldPoseLandmarks[POSE_LANDMARKS_LEFT.LEFT_FOOT_INDEX].pos,
                this.worldPoseLandmarks[POSE_LANDMARKS_LEFT.LEFT_ANKLE].pos
            ]),
            right: getBasis([
                this.worldPoseLandmarks[POSE_LANDMARKS_RIGHT.RIGHT_HEEL].pos,
                this.worldPoseLandmarks[POSE_LANDMARKS_RIGHT.RIGHT_FOOT_INDEX].pos,
                this.worldPoseLandmarks[POSE_LANDMARKS_RIGHT.RIGHT_ANKLE].pos
            ]),
        }

        for (const [k, v] of Object.entries(feet)) {
            const isLeft = k === 'left';
            const footBoneKey = `${k}Foot`;

            const thisBasis = v.negateAxes(AXIS.yz).transpose([1, 2, 0]);
            thisBasis.verifyBasis();

            // Root normal
            const footNormal = isLeft ? this.leftFootNormal : this.rightFootNormal;
            footNormal.copyFrom(thisBasis.z.negate());

            const thisFootRotation = this._boneRotations[footBoneKey];
            const basis1: Basis = thisFootRotation.baseBasis;
            const basis2 = thisBasis.rotateByQuaternion(this.applyQuaternionChain(footBoneKey, isLeft).conjugate());
            const thisFootBasisRotation = isLeft ? this.leftFootBasisRotation : this.rightFootBasisRotation;
            const footRotationQuaternionRaw = quaternionBetweenBases(
                basis1, basis2,
            );
            // TODO DEBUG
            const handNormalsKey = `${k}HandNormals`;
            const handNormals = this[handNormalsKey as PosesKey] as NormalizedLandmarkList;
            handNormals.length = 0;
            handNormals.push(vectorToNormalizedLandmark(thisBasis.x));
            handNormals.push(vectorToNormalizedLandmark(thisBasis.y));
            handNormals.push(vectorToNormalizedLandmark(thisBasis.z));

            const footRotationQuaternion = reverseRotation(footRotationQuaternionRaw, AXIS.yz);
            thisFootRotation.set(footRotationQuaternion);
            const footRotationDegrees = quaternionToDegrees(footRotationQuaternion);
        }
    }

    private preProcessResults() {
        // Preprocessing results
        // Create pose landmark list
        // @ts-ignore
        const inputWorldPoseLandmarks: NormalizedLandmarkList | undefined = this.cloneableInputResults?.ea;    // Seems to be the new pose_world_landmark
        const inputPoseLandmarks: NormalizedLandmarkList | undefined = this.cloneableInputResults?.poseLandmarks;    // Seems to be the new pose_world_landmark
        if (inputWorldPoseLandmarks && inputPoseLandmarks) {
            if (inputWorldPoseLandmarks.length !== POSE_LANDMARK_LENGTH)
                console.warn(`Pose Landmark list has a length less than ${POSE_LANDMARK_LENGTH}!`);

            this.inputPoseLandmarks = this.preProcessLandmarks(
                inputWorldPoseLandmarks, this.worldPoseLandmarks);
            this.preProcessLandmarks(
                inputPoseLandmarks, this.poseLandmarks);

            // Positional offset
            if ((inputWorldPoseLandmarks[POSE_LANDMARKS.LEFT_HIP].visibility || 0) > VISIBILITY_THRESHOLD &&
                (inputWorldPoseLandmarks[POSE_LANDMARKS.RIGHT_HIP].visibility || 0) > VISIBILITY_THRESHOLD
            ) {
                this.midHipOffset = vectorToNormalizedLandmark(
                    this.poseLandmarks[POSE_LANDMARKS.LEFT_HIP].pos
                        .add(this.poseLandmarks[POSE_LANDMARKS.RIGHT_HIP].pos)
                        .scaleInPlace(0.5)
                );
                this.midHipOffset.z = 0;    // No depth info
            }
        }

        const inputFaceLandmarks = this.cloneableInputResults?.faceLandmarks;    // Seems to be the new pose_world_landmark
        if (inputFaceLandmarks) {
            this.inputFaceLandmarks = this.preProcessLandmarks(
                inputFaceLandmarks, this.faceLandmarks);
        }

        // TODO: update wrist offset only when debugging
        const inputLeftHandLandmarks = this.cloneableInputResults?.leftHandLandmarks;
        const inputRightHandLandmarks = this.cloneableInputResults?.rightHandLandmarks;
        if (inputLeftHandLandmarks) {
            this.leftWristOffset.updatePosition(
                this.worldPoseLandmarks[POSE_LANDMARKS.LEFT_WRIST].pos.subtract(
                    normalizedLandmarkToVector(
                        inputLeftHandLandmarks[HAND_LANDMARKS.WRIST],
                        Poses.HAND_POSITION_SCALING,
                        true)
                )
            );
            this.inputLeftHandLandmarks = this.preProcessLandmarks(
                inputLeftHandLandmarks, this.leftHandLandmarks,
                this.leftWristOffset.pos, Poses.HAND_POSITION_SCALING);
        }
        if (inputRightHandLandmarks) {
            this.rightWristOffset.updatePosition(
                this.worldPoseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].pos.subtract(
                    normalizedLandmarkToVector(
                        inputRightHandLandmarks[HAND_LANDMARKS.WRIST],
                        Poses.HAND_POSITION_SCALING,
                        true)
                )
            );
            this.inputRightHandLandmarks = this.preProcessLandmarks(
                inputRightHandLandmarks, this.rightHandLandmarks,
                this.rightWristOffset.pos, Poses.HAND_POSITION_SCALING);
        }
    }

    private preProcessLandmarks(
        resultsLandmarks: NormalizedLandmark[],
        filteredLandmarks: FilteredVectorLandmarkList,
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
        // THUMB_CMC
        // THUMB_MCP
        // THUMB_IP
        for (let i = 1; i < 4; ++i) {
            const tMCP_X = new Vector3(isLeft ? 1 : -1, 0, -1.5).normalize();
            const tMCP_Y = new Vector3(0, isLeft ? -1 : 1, 0);
            const tMCP_Z = Vector3.Cross(tMCP_X, tMCP_Y).normalize();
            const basis = new Basis([
                tMCP_X,
                // new Vector3(0, 0, isLeft ? -1 : 1),
                tMCP_Y,
                tMCP_Z,
            ]).rotateByQuaternion(Quaternion.FromEulerAngles(0, 0, isLeft ? 0.2 : -0.2));
            this._boneRotations[handLandMarkToBoneName(i, isLeft)] =
                new CloneableQuaternion(
                    Quaternion.Identity(), basis);
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
        this._boneRotations['head'] = new CloneableQuaternion(
            Quaternion.Identity(), new Basis(null)
        );
        this._boneRotations['neck'] = new CloneableQuaternion(
            Quaternion.Identity(), new Basis(null)
        );
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
                Quaternion.FromEulerAngles(0, 0, isLeft ? 1.0472 : -1.0472),
                new Basis([
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
        for (const k of lr) {
            const isLeft = k === "left";
            this._boneRotations[`${k}UpperLeg`] = new CloneableQuaternion(
                Quaternion.Identity(), new Basis([
                    new Vector3(0, -1, 0),
                    new Vector3(-1, 0, 0),
                    new Vector3(0, 0, -1),
                ]).rotateByQuaternion(Quaternion.FromEulerAngles(
                    0, 0, isLeft ? -0.05236 : 0.05236)));
            this._boneRotations[`${k}LowerLeg`] = new CloneableQuaternion(
                Quaternion.Identity(), new Basis([
                    new Vector3(0, -1, 0),
                    new Vector3(-1, 0, 0),
                    new Vector3(0, 0, -1),
                ]).rotateByQuaternion(Quaternion.FromEulerAngles(
                    0, 0, isLeft ? -0.0873 : 0.0873)));
        }
        // Feet
        for (const k of lr) {
            const isLeft = k === "left";
            const startBasis = new Basis([
                new Vector3(0, -1, 0),
                new Vector3(-1, 0, 0),
                new Vector3(0, 0, -1),
            ]);
            const rX = Quaternion.RotationAxis(startBasis.x.clone(), isLeft ? -0.2618 : 0.2618);
            const z1 = Vector3.Zero();
            startBasis.z.rotateByQuaternionToRef(rX, z1);
            const rZ = Quaternion.RotationAxis(z1, isLeft ? 0.0873 : -0.0873);
            const thisFootBasisRotation = isLeft ? this.leftFootBasisRotation : this.rightFootBasisRotation;
            thisFootBasisRotation.copyFrom(rX.multiply(rZ));

            this._boneRotations[`${k}Foot`] = new CloneableQuaternion(
                Quaternion.Identity(), startBasis);
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

    private shallUpdateArm(isLeft: boolean) {
        // Update only when all leg landmarks are visible
        const shoulderVisilibity = this.cloneableInputResults?.poseLandmarks[
            isLeft ? POSE_LANDMARKS.LEFT_SHOULDER : POSE_LANDMARKS.RIGHT_SHOULDER].visibility || 0;
        const elbowVisilibity = this.cloneableInputResults?.poseLandmarks[
            isLeft ? POSE_LANDMARKS.LEFT_ELBOW : POSE_LANDMARKS.RIGHT_ELBOW].visibility || 0;
        const wristVisilibity = this.cloneableInputResults?.poseLandmarks[
            isLeft ? POSE_LANDMARKS.LEFT_WRIST : POSE_LANDMARKS.RIGHT_WRIST].visibility || 0;
        return !(shoulderVisilibity <= VISIBILITY_THRESHOLD || elbowVisilibity <= VISIBILITY_THRESHOLD
            || wristVisilibity <= VISIBILITY_THRESHOLD);

    }

    private shallUpdateLegs() {
        for (const isLeft of [true, false]) {
            // Update only when all leg landmarks are visible
            const kneeVisilibity = this.cloneableInputResults?.poseLandmarks[
                isLeft ? POSE_LANDMARKS_LEFT.LEFT_KNEE : POSE_LANDMARKS_RIGHT.RIGHT_KNEE].visibility || 0;
            const ankleVisilibity = this.cloneableInputResults?.poseLandmarks[
                isLeft ? POSE_LANDMARKS_LEFT.LEFT_ANKLE : POSE_LANDMARKS_RIGHT.RIGHT_ANKLE].visibility || 0;
            const footVisilibity = this.cloneableInputResults?.poseLandmarks[
                isLeft ? POSE_LANDMARKS_LEFT.LEFT_FOOT_INDEX : POSE_LANDMARKS_RIGHT.RIGHT_FOOT_INDEX].visibility || 0;
            const heelVisilibity = this.cloneableInputResults?.poseLandmarks[
                isLeft ? POSE_LANDMARKS_LEFT.LEFT_HEEL : POSE_LANDMARKS_RIGHT.RIGHT_HEEL].visibility || 0;
            if (kneeVisilibity <= VISIBILITY_THRESHOLD || ankleVisilibity <= VISIBILITY_THRESHOLD
                || footVisilibity <= VISIBILITY_THRESHOLD || heelVisilibity <= VISIBILITY_THRESHOLD)
                return false;
        }
        return true;
    }
}

const poseResults : Poses = new Poses();

function sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

Comlink.expose(poseResults);

export {poseResults};
