/*
Copyright (C) 2022  The v3d Authors.

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

import {NormalizedLandmark, Results} from "@mediapipe/holistic";
import {Nullable, Vector3} from "@babylonjs/core";
import {GaussianVectorFilter, KalmanVectorFilter, OneEuroVectorFilter, VISIBILITY_THRESHOLD} from "./filter";
import {objectFlip} from "./utils";


export interface FilteredLandmarkParams {
    R?: number,
    Q?: number,
    oneEuroCutoff?: number,
    oneEuroBeta?: number,
    type: string,
    gaussianSigma?: number,
}

export class FilteredVectorLandmark {
    private mainFilter: OneEuroVectorFilter | KalmanVectorFilter;
    private gaussianVectorFilter: Nullable<GaussianVectorFilter> = null;

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
        params: FilteredLandmarkParams = {
            oneEuroCutoff: 0.01,
            oneEuroBeta: 0,
            type: 'OneEuro'
        }
    ) {
        if (params.type === "Kalman")
            this.mainFilter = new KalmanVectorFilter(params.R, params.Q);
        else if (params.type === "OneEuro")
            this.mainFilter = new OneEuroVectorFilter(
                this.t,
                this.pos,
                Vector3.Zero(),
                params.oneEuroCutoff,
                params.oneEuroBeta);
        else
            throw Error("Wrong filter type!");
        if (params.gaussianSigma)
            this.gaussianVectorFilter = new GaussianVectorFilter(5, params.gaussianSigma);
    }

    public updatePosition(pos: Vector3, visibility?: number) {
        this.t += 1;

        // Face Mesh has no visibility
        if (visibility === undefined || visibility > VISIBILITY_THRESHOLD) {
            pos = this.mainFilter.next(this.t, pos);

            if (this.gaussianVectorFilter) {
                this.gaussianVectorFilter.push(pos);
                pos = this.gaussianVectorFilter.apply();
            }

            this._pos = pos;

            this.visibility = visibility;
        }
    }
}

export type FilteredVectorLandmarkList = FilteredVectorLandmark[];

export type FilteredVectorLandmark3 = [
    FilteredVectorLandmark,
    FilteredVectorLandmark,
    FilteredVectorLandmark,
];
export interface CloneableResults extends Omit<Results, 'segmentationMask'|'image'> {}

export const POSE_LANDMARK_LENGTH = 33;
export const FACE_LANDMARK_LENGTH = 478;
export const HAND_LANDMARK_LENGTH = 21;

export const normalizedLandmarkToVector = (
    l: NormalizedLandmark,
    scaling = 1.,
    reverseY = false) => {
    return new Vector3(
        l.x * scaling,
        reverseY ? -l.y * scaling : l.y * scaling,
        l.z * scaling);
}
export const vectorToNormalizedLandmark = (l: Vector3) : NormalizedLandmark => {
    return {x: l.x, y: l.y, z: l.z};
};

export const HAND_LANDMARKS = {
    WRIST: 0,
    THUMB_CMC: 1,
    THUMB_MCP: 2,
    THUMB_IP: 3,
    THUMB_TIP: 4,
    INDEX_FINGER_MCP: 5,
    INDEX_FINGER_PIP: 6,
    INDEX_FINGER_DIP: 7,
    INDEX_FINGER_TIP: 8,
    MIDDLE_FINGER_MCP: 9,
    MIDDLE_FINGER_PIP: 10,
    MIDDLE_FINGER_DIP: 11,
    MIDDLE_FINGER_TIP: 12,
    RING_FINGER_MCP: 13,
    RING_FINGER_PIP: 14,
    RING_FINGER_DIP: 15,
    RING_FINGER_TIP: 16,
    PINKY_MCP: 17,
    PINKY_PIP: 18,
    PINKY_DIP: 19,
    PINKY_TIP: 20,
};

export const HAND_LANDMARKS_BONE_MAPPING = {
    Hand: HAND_LANDMARKS.WRIST,
    ThumbProximal: HAND_LANDMARKS.THUMB_CMC,
    ThumbIntermediate: HAND_LANDMARKS.THUMB_MCP,
    ThumbDistal: HAND_LANDMARKS.THUMB_IP,
    IndexProximal: HAND_LANDMARKS.INDEX_FINGER_MCP,
    IndexIntermediate: HAND_LANDMARKS.INDEX_FINGER_PIP,
    IndexDistal: HAND_LANDMARKS.INDEX_FINGER_DIP,
    MiddleProximal: HAND_LANDMARKS.MIDDLE_FINGER_MCP,
    MiddleIntermediate: HAND_LANDMARKS.MIDDLE_FINGER_PIP,
    MiddleDistal: HAND_LANDMARKS.MIDDLE_FINGER_DIP,
    RingProximal: HAND_LANDMARKS.RING_FINGER_MCP,
    RingIntermediate: HAND_LANDMARKS.RING_FINGER_PIP,
    RingDistal: HAND_LANDMARKS.RING_FINGER_DIP,
    LittleProximal: HAND_LANDMARKS.PINKY_MCP,
    LittleIntermediate: HAND_LANDMARKS.PINKY_PIP,
    LittleDistal: HAND_LANDMARKS.PINKY_DIP,
};
export const HAND_LANDMARKS_BONE_REVERSE_MAPPING: {[key:number] : string} = objectFlip(HAND_LANDMARKS_BONE_MAPPING);
export type HandBoneMappingKey = keyof typeof HAND_LANDMARKS_BONE_MAPPING;
export function handLandMarkToBoneName(landmark: number, isLeft: boolean) {
    if (!(landmark in HAND_LANDMARKS_BONE_REVERSE_MAPPING)) throw Error("Wrong landmark given!");
    return (isLeft ? 'left' : 'right') + HAND_LANDMARKS_BONE_REVERSE_MAPPING[landmark];
}

/*
 * Depth-first search/walk of a generic tree.
 * Also returns a map for backtracking from leaf.
 */
export function depthFirstSearch(
    rootNode: any,
    f: (n: any) => boolean
): [any, any] {
    const stack = [];
    const parentMap: Map<any, any> = new Map<any, any>();
    stack.push(rootNode);

    while (stack.length !== 0) {
        // remove the first child in the stack
        const currentNode: any = stack.splice(-1, 1)[0];
        const retVal = f(currentNode);
        if (retVal) return [currentNode, parentMap];

        const currentChildren = currentNode.children;
        // add any children in the node at the top of the stack
        if (currentChildren !== null) {
            for (let index = 0; index < currentChildren.length; index++) {
                const child = currentChildren[index];
                stack.push(child);
                if (!(parentMap.has(child))) {
                    parentMap.set(child, currentNode);
                }
            }
        }
    }
    return [null, null];
}
