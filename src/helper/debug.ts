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

import {Mesh, MeshBuilder, Nullable, Quaternion, Scene, StandardMaterial, Vector3} from "@babylonjs/core";
import {Color3, Vector4} from "@babylonjs/core/Maths";
import {
    Arrow3D,
    cloneableQuaternionToQuaternion,
    HAND_LANDMARK_LENGTH,
    initArray,
    normalizedLandmarkToVector,
    POSE_LANDMARK_LENGTH
} from "./utils";
import chroma from "chroma-js";
import {NormalizedLandmark, NormalizedLandmarkList, POSE_LANDMARKS} from "@mediapipe/holistic";
import {CloneableQuaternion, CloneableQuaternionList, HandBoneRotations} from "../worker/pose-processing";

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

export function makeSphere(
    scene: Scene,
    pos?: Vector3,
    color?: number | string,
    options?: createSphereOptions) : Mesh {
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
    let result = Vector3.Zero();
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

    constructor(
        private readonly scene: Scene
    ) {
        this.poseLandmarkSpheres = this.initLandmarks(POSE_LANDMARK_LENGTH);
        this.faceNormalArrows = this.initNormalArrows(1);
        this.leftHandLandmarkSpheres = this.initLandmarks(HAND_LANDMARK_LENGTH, '#ff0000');
        this.rightHandLandmarkSpheres = this.initLandmarks(HAND_LANDMARK_LENGTH, '#0022ff');
        this.irisNormalArrows = this.initIrisNormalArrows();

        this.leftHandNormalArrow = new Arrow3D(this.scene,
            0.02, 32, 0.06, 0.06,
            0.5, Vector3.Zero(), Vector3.One());
        this.rightHandNormalArrow = new Arrow3D(this.scene,
            0.02, 32, 0.06, 0.06,
            0.5, Vector3.Zero(), Vector3.One());

        this.leftHandNormalArrows = this.initNormalArrows(4);
        this.rightHandNormalArrows = this.initNormalArrows(4);

        scene.debugLayer.show({
            globalRoot: document.getElementById('wrapper') as HTMLElement,
            handleResize: true,
        });
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
        return initArray<Arrow3D>(
            length,    // Temp magical number
            () => new Arrow3D(this.scene,
                0.02, 32, 0.08, 0.08,
                0.5, Vector3.Zero(), Vector3.One()));
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
            [],[],
            [9, 4, 15, 12], [0, 4, 8, 12],
            [0, 1, 2, 3], [0, 1, 2, 3],
            [24, 25, 26, 34, 35, 36],
        ];
        if (resultFaceMeshIndexLandmarks.length !== 0 && !this.faceMeshLandmarkSpheres)
            this.faceMeshLandmarkSpheres = this.initFaceMeshLandmarks(resultFaceMeshIndexLandmarks);
        if (!this.faceMeshLandmarkSpheres ||
            resultFaceMeshLandmarks.length !== this.faceMeshLandmarkSpheres.length) return;
        for (let i = 0; i < this.faceMeshLandmarkSpheres.length; ++i) {
            for (let j = 0; j < this.faceMeshLandmarkSpheres[i].length; ++j) {
                if (!toShow[i].includes(j)) continue;
                this.faceMeshLandmarkSpheres[i][j].position.set(
                    resultFaceMeshLandmarks[i][j].x,
                    resultFaceMeshLandmarks[i][j].y,
                    resultFaceMeshLandmarks[i][j].z
                );
            }
        }
    }

    public updateHandWristNormalArrows(
        resultLeftHandBoneRotations: HandBoneRotations,
        resultRightHandBoneRotations: HandBoneRotations,
        resultPoseLandmarks: NormalizedLandmarkList
    ) {
        const baseRootNormal = new Vector3(0, -1, 0);
        this.leftHandNormalArrow.updateStartAndDirection(
            normalizedLandmarkToVector(
                resultPoseLandmarks[POSE_LANDMARKS.LEFT_WRIST]),
            quaternionToDirectionVector(
                baseRootNormal, resultLeftHandBoneRotations.Hand),
        );
        this.rightHandNormalArrow.updateStartAndDirection(
            normalizedLandmarkToVector(
                resultPoseLandmarks[POSE_LANDMARKS.RIGHT_WRIST]),
            quaternionToDirectionVector(
                baseRootNormal, resultRightHandBoneRotations.Hand),
        );
    }

    public updateHandNormalArrows(
        resultLeftHandNormals: NormalizedLandmarkList,
        resultRightHandNormals: NormalizedLandmarkList,
        resultPoseLandmarks: NormalizedLandmarkList
    ) {
        if (resultLeftHandNormals.length !== this.leftHandNormalArrows.length
            || resultRightHandNormals.length !== this.rightHandNormalArrows.length) return;
        for (let i = 0; i < this.leftHandNormalArrows.length; ++i) {
            this.leftHandNormalArrows[i].updateStartAndDirection(
                normalizedLandmarkToVector(
                    resultPoseLandmarks[POSE_LANDMARKS.LEFT_WRIST]),
                normalizedLandmarkToVector(resultLeftHandNormals[i]),
            );
        }
        for (let i = 0; i < this.rightHandNormalArrows.length; ++i) {
            this.rightHandNormalArrows[i].updateStartAndDirection(
                normalizedLandmarkToVector(
                    resultPoseLandmarks[POSE_LANDMARKS.RIGHT_WRIST]),
                normalizedLandmarkToVector(resultRightHandNormals[i]),
            );
        }
    }

}
