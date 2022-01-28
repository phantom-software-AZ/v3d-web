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

import {Mesh, MeshBuilder, Nullable, Scene, StandardMaterial, Vector3} from "@babylonjs/core";
import {Color3, Vector4} from "@babylonjs/core/Maths";
import {
    initArray,
} from "./utils";
import chroma from "chroma-js";
import {NormalizedLandmark, NormalizedLandmarkList, POSE_LANDMARKS} from "@mediapipe/holistic";
import {Poses} from "../worker/pose-processing";
import {CloneableQuaternion, CloneableQuaternionList, cloneableQuaternionToQuaternion} from "./quaternion";
import {HAND_LANDMARK_LENGTH, HAND_LANDMARKS, normalizedLandmarkToVector, POSE_LANDMARK_LENGTH} from "./landmark";

type createSphereOptions = {
    segments?: number;
    diameter?: number;
    diameterX?: number;
    diameterY?: number;
    diameterZ?: number;
    arc?: number;
    slice?: number;
    sideOrientation?: number;
    frontUVs?: Vector4;
    backUVs?: Vector4;
    updatable?: boolean;
};

export class Arrow3D {
    //Shape profile in XY plane
    private readonly myShape: Vector3[] = [];
    private readonly myPath: Vector3[] = [];
    private arrowInstance: Nullable<Mesh> = null;

    private _arrowRadius: number;
    get arrowRadius(): number {
        return this._arrowRadius;
    }

    set arrowRadius(value: number) {
        this._arrowRadius = value;
        this.updateTopShape();
    }

    private _n: number;
    get n(): number {
        return this._n;
    }

    set n(value: number) {
        this._n = value;
        this.updateTopShape();
    }

    private _arrowHeadLength: number;
    get arrowHeadLength(): number {
        return this._arrowHeadLength;
    }

    set arrowHeadLength(value: number) {
        this._arrowHeadLength = value;
        this.updatePath();
    }

    private _arrowHeadMaxSize: number;

    get arrowStart(): Vector3 {
        return this._arrowStart;
    }

    set arrowStart(value: Vector3) {
        this._arrowStart = value;
        this.updatePath();
    }

    private _arrowLength: number;
    get arrowLength(): number {
        return this._arrowLength;
    }

    set arrowLength(value: number) {
        this._arrowLength = value;
        this.updatePath();
    }

    private _arrowStart: Vector3;

    get arrowHeadMaxSize(): number {
        return this._arrowHeadMaxSize;
    }

    set arrowHeadMaxSize(value: number) {
        this._arrowHeadMaxSize = value;
        this.updatePath();
    }

    private _arrowDirection: Vector3;
    get arrowDirection(): Vector3 {
        return this._arrowDirection;
    }

    set arrowDirection(value: Vector3) {
        this._arrowDirection = value;
        this.updatePath();
    }

    private _material: StandardMaterial;
    get material(): StandardMaterial {
        return this._material;
    }

    constructor(
        private scene: Scene,
        arrowRadius = 0.5,
        n = 30,
        arrowHeadLength = 1.5,
        arrowHeadMaxSize = 1.5,
        arrowLength = 10,
        arrowStart: Vector3,
        arrowDirection: Vector3,
        color?: number | string,
    ) {
        this._arrowRadius = arrowRadius;
        this._n = n;
        this._arrowHeadLength = arrowHeadLength;
        this._arrowHeadMaxSize = arrowHeadMaxSize;
        this._arrowLength = arrowLength;
        this._arrowStart = arrowStart;
        this._arrowDirection = arrowDirection;
        this.updateTopShape();
        this.updatePath();
        this._material = new StandardMaterial("sphereMaterial", scene);
        if (color) {
            if (typeof color === 'number') color = `#${color.toString(16)}`;
            this._material.diffuseColor = Color3.FromHexString(color);
        }
    }

    private updateTopShape() {
        const deltaAngle = 2 * Math.PI / this.n;
        this.myShape.length = 0;
        for (let i = 0; i <= this.n; i++) {
            this.myShape.push(new Vector3(
                this.arrowRadius * Math.cos(i * deltaAngle),
                this.arrowRadius * Math.sin(i * deltaAngle),
                0))
        }
        this.myShape.push(this.myShape[0]);  //close profile
    }

    private updatePath() {
        const arrowBodyLength = this.arrowLength - this.arrowHeadLength;
        this.arrowDirection.normalize();
        const arrowBodyEnd = this.arrowStart.add(this.arrowDirection.scale(arrowBodyLength));
        const arrowHeadEnd = arrowBodyEnd.add(this.arrowDirection.scale(this.arrowHeadLength));

        this.myPath.length = 0;
        this.myPath.push(this.arrowStart);
        this.myPath.push(arrowBodyEnd);
        this.myPath.push(arrowBodyEnd);
        this.myPath.push(arrowHeadEnd);

        if (!this.arrowInstance)
            this.arrowInstance = MeshBuilder.ExtrudeShapeCustom(
                "arrow",
                {
                    shape: this.myShape,
                    path: this.myPath,
                    updatable: true,
                    scaleFunction: this.scaling.bind(this),
                    sideOrientation: Mesh.DOUBLESIDE
                }, this.scene);
        else
            this.arrowInstance = MeshBuilder.ExtrudeShapeCustom(
                "arrow",
                {
                    shape: this.myShape,
                    path: this.myPath,
                    scaleFunction: this.scaling.bind(this),
                    instance: this.arrowInstance
                }, this.scene);
        this.arrowInstance.material = this.material;
    }

    private scaling(index: number, distance: number): number {
        switch (index) {
            case 0:
            case 1:
                return 1;
            case 2:
                return this.arrowHeadMaxSize / this.arrowRadius;
            case 3:
                return 0;
            default:
                return 1;
        }
    }

    public updateStartAndDirection(arrowStart: Vector3, arrowDirection: Vector3) {
        this._arrowStart = arrowStart;
        this._arrowDirection = arrowDirection.length() === 0 ?
            Vector3.One() : arrowDirection;
        this.updatePath();
    }
}

export function makeSphere(
    scene: Scene,
    pos?: Vector3,
    color?: number | string,
    options?: createSphereOptions): Mesh {
    const sphere = MeshBuilder.CreateSphere("sphere",
        options || {
            diameterX: 1, diameterY: 0.5, diameterZ: 0.5
        }, scene);
    const material = new StandardMaterial("sphereMaterial", scene);
    if (color) {
        if (typeof color === 'number') color = `#${color.toString(16)}`;
        material.diffuseColor = Color3.FromHexString(color);
    }
    sphere.material = material;

    if (pos)
        sphere.position = pos;

    return sphere;
}

export function quaternionToDirectionVector(
    base: Vector3,
    resultQuaternion: CloneableQuaternion
): Vector3 {
    const quaternion = cloneableQuaternionToQuaternion(resultQuaternion);
    const result = Vector3.Zero();
    base.rotateByQuaternionToRef(quaternion, result);
    return result.normalize();
}

export class DebugInfo {
    private readonly poseLandmarkSpheres: Mesh[];
    private readonly faceNormalArrows: Arrow3D[];
    private faceMeshLandmarkSpheres: Nullable<Mesh[][]> = null;
    private readonly leftHandLandmarkSpheres: Mesh[];
    private readonly rightHandLandmarkSpheres: Mesh[];
    private readonly irisNormalArrows: Arrow3D[];
    private readonly leftHandNormalArrow: Arrow3D;
    private readonly rightHandNormalArrow: Arrow3D;
    private readonly leftHandNormalArrows: Arrow3D[];
    private readonly rightHandNormalArrows: Arrow3D[];
    private readonly poseNormalArrows: Arrow3D[];

    constructor(
        private readonly scene: Scene
    ) {
        this.poseLandmarkSpheres = this.initLandmarks(POSE_LANDMARK_LENGTH);
        this.faceNormalArrows = this.initNormalArrows(1);
        this.leftHandLandmarkSpheres = this.initLandmarks(HAND_LANDMARK_LENGTH, '#ff0000');
        this.rightHandLandmarkSpheres = this.initLandmarks(HAND_LANDMARK_LENGTH, '#0022ff');
        this.irisNormalArrows = this.initIrisNormalArrows();
        this.poseNormalArrows = this.initNormalArrows(3);

        this.leftHandNormalArrow = new Arrow3D(this.scene,
            0.02, 32, 0.06, 0.06,
            0.5, Vector3.Zero(), Vector3.One());
        this.rightHandNormalArrow = new Arrow3D(this.scene,
            0.02, 32, 0.06, 0.06,
            0.5, Vector3.Zero(), Vector3.One());

        this.leftHandNormalArrows = this.initNormalArrows(6);
        this.rightHandNormalArrows = this.initNormalArrows(6);

        if (!scene.debugLayer.isVisible()) {
            scene.debugLayer.show({
                globalRoot: document.getElementById('wrapper') as HTMLElement,
                handleResize: true,
            });
        }
    }

    private initLandmarks(
        length: number,
        color?: number | string
    ) {
        const colors = chroma.scale('Spectral')
            .colors(length, 'hex');
        return initArray<Mesh>(
            length,
            (i) => makeSphere(
                this.scene, Vector3.One(), colors[i], {diameter: 0.03}));
    }

    private initNormalArrows(length: number) {
        const colors = chroma.scale('Spectral')
            .colors(length / 2, 'hex');
        return initArray<Arrow3D>(
            length,    // Temp magical number
            (i) => new Arrow3D(this.scene,
                0.01, 32, 0.02, 0.02,
                0.2, Vector3.Zero(), Vector3.One(), colors[i % (length / 2)]));
    }

    private initIrisNormalArrows() {
        return initArray<Arrow3D>(
            2,
            () => new Arrow3D(this.scene,
                0.01, 32, 0.04, 0.04,
                0.25, Vector3.Zero(), Vector3.One()));
    }

    private initFaceMeshLandmarks(indexList: number[][]) {
        return initArray<Mesh[]>(
            indexList.length,
            (i) => {
                return initArray<Mesh>(
                    indexList[i].length,
                    ((() => {
                        const colors = chroma.scale('Spectral')
                            .colors(indexList[i].length, 'hex');
                        return (i) => {
                            return makeSphere(
                                this.scene, Vector3.One(), colors[i], {diameter: 0.01})
                        };
                    })()));
            });
    }

    public updatePoseLandmarkSpheres(resultPoseLandmarks: NormalizedLandmarkList) {
        if (resultPoseLandmarks.length !== POSE_LANDMARK_LENGTH) return;
        for (let i = 0; i < POSE_LANDMARK_LENGTH; ++i) {
            if ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22].includes(i)) continue;
            this.poseLandmarkSpheres[i].position.set(
                resultPoseLandmarks[i].x,
                resultPoseLandmarks[i].y,
                resultPoseLandmarks[i].z
            );
        }
    }

    public updateHandLandmarkSpheres(
        resultHandLandmarks: NormalizedLandmarkList,
        left: boolean
    ) {
        const landmarkSpheres = left ?
            this.leftHandLandmarkSpheres :
            this.rightHandLandmarkSpheres;
        if (resultHandLandmarks.length !== HAND_LANDMARK_LENGTH) return;
        for (let i = 0; i < HAND_LANDMARK_LENGTH; ++i) {
            landmarkSpheres[i].position.set(
                resultHandLandmarks[i].x,
                resultHandLandmarks[i].y,
                resultHandLandmarks[i].z
            );
        }
    }

    public updateFaceNormalArrows(
        resultFaceNormals: NormalizedLandmarkList,
        resultPoseLandmarks: NormalizedLandmarkList
    ) {
        if (resultFaceNormals.length !== this.faceNormalArrows.length) return;
        for (let i = 0; i < this.faceNormalArrows.length; ++i) {
            this.faceNormalArrows[i].updateStartAndDirection(
                normalizedLandmarkToVector(
                    resultPoseLandmarks[POSE_LANDMARKS.NOSE]),
                normalizedLandmarkToVector(resultFaceNormals[i]),
            );
        }
    }

    public updateIrisQuaternionArrows(
        resultIrisQuaternions: CloneableQuaternionList,
        resultPoseLandmarks: NormalizedLandmarkList,
        resultFaceNormal: NormalizedLandmark
    ) {
        if (resultIrisQuaternions.length !== 2 && resultIrisQuaternions.length !== 3) return;
        this.irisNormalArrows[0].updateStartAndDirection(
            normalizedLandmarkToVector(
                resultPoseLandmarks[POSE_LANDMARKS.LEFT_EYE]),
            quaternionToDirectionVector(
                normalizedLandmarkToVector(resultFaceNormal),
                resultIrisQuaternions[0]),
        );
        this.irisNormalArrows[1].updateStartAndDirection(
            normalizedLandmarkToVector(
                resultPoseLandmarks[POSE_LANDMARKS.RIGHT_EYE]),
            quaternionToDirectionVector(
                normalizedLandmarkToVector(resultFaceNormal),
                resultIrisQuaternions[1]),
        );
    }

    public updateFaceMeshLandmarkSpheres(
        resultFaceMeshIndexLandmarks: number[][],
        resultFaceMeshLandmarks: NormalizedLandmarkList[]) {
        const toShow = [
            [0], [0],
            [9, 4, 15, 12], [0, 4, 8, 12],
            [0], [0],
            [24, 25, 26, 34, 35, 36],
            [0, 6, 18, 30]
        ];
        if (resultFaceMeshIndexLandmarks.length !== 0 && !this.faceMeshLandmarkSpheres)
            this.faceMeshLandmarkSpheres = this.initFaceMeshLandmarks(resultFaceMeshIndexLandmarks);
        if (!this.faceMeshLandmarkSpheres ||
            resultFaceMeshLandmarks.length !== this.faceMeshLandmarkSpheres.length) return;
        for (let i = 0; i < this.faceMeshLandmarkSpheres.length; ++i) {
            for (let j = 0; j < this.faceMeshLandmarkSpheres[i].length; ++j) {
                if (toShow[i].length > 0 && !toShow[i].includes(j)) continue;
                this.faceMeshLandmarkSpheres[i][j].position.set(
                    resultFaceMeshLandmarks[i][j].x,
                    resultFaceMeshLandmarks[i][j].y,
                    resultFaceMeshLandmarks[i][j].z
                );
            }
        }
    }

    public updateHandWristNormalArrows(
        resultLeftHandBoneRotations: CloneableQuaternionList,
        resultRightHandBoneRotations: CloneableQuaternionList,
        resultPoseLandmarks: NormalizedLandmarkList
    ) {
        this.leftHandNormalArrow.updateStartAndDirection(
            normalizedLandmarkToVector(
                resultPoseLandmarks[POSE_LANDMARKS.LEFT_WRIST]),
            quaternionToDirectionVector(
                Poses.HAND_BASE_ROOT_NORMAL, resultLeftHandBoneRotations[HAND_LANDMARKS.WRIST]),
        );
        this.rightHandNormalArrow.updateStartAndDirection(
            normalizedLandmarkToVector(
                resultPoseLandmarks[POSE_LANDMARKS.RIGHT_WRIST]),
            quaternionToDirectionVector(
                Poses.HAND_BASE_ROOT_NORMAL, resultRightHandBoneRotations[HAND_LANDMARKS.WRIST]),
        );
    }

    public updateHandNormalArrows(
        resultLeftHandNormals: Nullable<NormalizedLandmarkList>,
        resultRightHandNormals: Nullable<NormalizedLandmarkList>,
        resultPoseLandmarks: NormalizedLandmarkList
    ) {
        if (resultLeftHandNormals) {
            for (let i = 0; i < Math.min(this.leftHandNormalArrows.length,
                resultLeftHandNormals.length); ++i) {
                this.leftHandNormalArrows[i].updateStartAndDirection(
                    // normalizedLandmarkToVector(
                    //     resultPoseLandmarks[POSE_LANDMARKS.LEFT_WRIST]),
                    i < 3 ? Vector3.Zero() : new Vector3(0, 1, 0),
                    normalizedLandmarkToVector(resultLeftHandNormals[i]),
                );
            }
        }
        if (resultRightHandNormals) {
            for (let i = 0; i < Math.min(this.rightHandNormalArrows.length,
                resultRightHandNormals.length); ++i) {
                this.rightHandNormalArrows[i].updateStartAndDirection(
                    // normalizedLandmarkToVector(
                    //     resultPoseLandmarks[POSE_LANDMARKS.RIGHT_WRIST]),
                    i < 3 ? Vector3.One() : new Vector3(0, 1, 0),
                    normalizedLandmarkToVector(resultRightHandNormals[i]),
                );
            }
        }
    }

    public updatePoseNormalArrows(
        resultPoseNormals: NormalizedLandmarkList,
        resultPoseLandmarks: NormalizedLandmarkList
    ) {
        if (resultPoseNormals.length !== this.poseNormalArrows.length) return;
        for (let i = 0; i < this.poseNormalArrows.length; ++i) {
            this.poseNormalArrows[i].updateStartAndDirection(
                // normalizedLandmarkToVector(
                //     resultPoseLandmarks[POSE_LANDMARKS.NOSE]),
                Vector3.Zero(),
                normalizedLandmarkToVector(resultPoseNormals[i]),
            );
        }
    }
}
