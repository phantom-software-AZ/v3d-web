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

import {
    Angle,
    StandardMaterial,
    Color3,
    Matrix,
    Mesh,
    MeshBuilder,
    Nullable,
    Plane,
    PrecisionDate,
    Quaternion,
    Scene,
    Vector3
} from "@babylonjs/core";
import {NormalizedLandmark} from "@mediapipe/holistic";
import {CloneableQuaternion} from "../worker/pose-processing";
import KalmanFilter from 'kalmanjs';

export function initArray<T>(length: number, initializer: (i: number) => T) {
    let arr = new Array<T>(length);
    for (let i = 0; i < length; i++)
        arr[i] = initializer(i);
    return arr;
}


export const gaussianKernel1d = (function () {
    let sqr2pi = Math.sqrt(2 * Math.PI);

    return function gaussianKernel1d (size: number, sigma: number) {
        // ensure size is even and prepare variables
        let width = (size / 2) | 0,
            kernel = new Array(width * 2 + 1),
            norm = 1.0 / (sqr2pi * sigma),
            coefficient = 2 * sigma * sigma,
            total = 0,
            x;

        // set values and increment total
        for (x = -width; x <= width; x++) {
            total += kernel[width + x] = norm * Math.exp(-x * x / coefficient);
        }

        // divide by total to make sure the sum of all the values is equal to 1
        for (x = 0; x < kernel.length; x++) {
            kernel[x] /= total;
        }

        return kernel;
    };
}());

/*
 * Converted from https://github.com/jaantollander/OneEuroFilter.
 */
export class OneEuroVectorFilter {
    constructor(
        public t_prev: number,
        public x_prev: Vector3,
        private dx_prev = Vector3.Zero(),
        public min_cutoff = 1.0,
        public beta = 0.0,
        public d_cutoff = 1.0
    ) {
    }

    private static smoothing_factor(t_e: number, cutoff: number) {
        const r = 2 * Math.PI * cutoff * t_e;
        return r / (r + 1);
    }

    private static exponential_smoothing(a: number, x: Vector3, x_prev: Vector3) {
        return x.scale(a).addInPlace(x_prev.scale((1 - a)));
    }

    public next(t: number, x: Vector3) {
        const t_e = t - this.t_prev;

        // The filtered derivative of the signal.
        const a_d = OneEuroVectorFilter.smoothing_factor(t_e, this.d_cutoff);
        const dx = x.subtract(this.x_prev).scaleInPlace(1 / t_e);
        const dx_hat = OneEuroVectorFilter.exponential_smoothing(a_d, dx, this.dx_prev);

        // The filtered signal.
        const cutoff = this.min_cutoff + this.beta * dx_hat.length();
        const a = OneEuroVectorFilter.smoothing_factor(t_e, cutoff);
        const x_hat = OneEuroVectorFilter.exponential_smoothing(a, x, this.x_prev);

        // Memorize the previous values.
        this.x_prev = x_hat;
        this.dx_prev = dx_hat;
        this.t_prev = t;

        return x_hat;
    }
}
export class KalmanVectorFilter {
    private readonly kalmanFilterX;
    private readonly kalmanFilterY;
    private readonly kalmanFilterZ;
    constructor(
        public R = 0.1,
        public Q = 3,
    ) {
        this.kalmanFilterX = new KalmanFilter({Q: Q, R: R});
        this.kalmanFilterY = new KalmanFilter({Q: Q, R: R});
        this.kalmanFilterZ = new KalmanFilter({Q: Q, R: R});
    }

    public next(t: number, vec: Vector3) {
        const newValues = [
            this.kalmanFilterX.filter(vec.x),
            this.kalmanFilterY.filter(vec.y),
            this.kalmanFilterZ.filter(vec.z),
        ]

        return Vector3.FromArray(newValues);
    }
}

export class GaussianVectorFilter {
    private _values: Vector3[] = [];
    get values(): Vector3[] {
        return this._values;
    }
    private readonly kernel: number[];

    constructor(
        public readonly size: number,
        private readonly sigma: number
    ) {
        if (size < 2) throw RangeError("Filter size too short");
        this.size = Math.floor(size);
        this.kernel = gaussianKernel1d(size, sigma);
    }

    public push(v: Vector3) {
        this.values.push(v);

        if (this.values.length === this.size + 1) {
            this.values.shift();
        } else if (this.values.length > this.size + 1) {
            console.warn(`Internal queue has length longer than size: ${this.size}`);
            this.values.slice(-this.size);
        }
    }

    public reset() {
        this.values.length = 0;
    }

    public apply() {
        if (this.values.length !== this.size) return Vector3.Zero();
        const ret = this.values[0].clone();
        const len0 = ret.length();
        for (let i = 0; i < this.size; ++i) {
            ret.addInPlace(this.values[i].scale(this.kernel[i]));
        }
        const len1 = ret.length();
        // Normalize to original length
        ret.scaleInPlace(len0 / len1);

        return ret;
    }
}

export class EuclideanHighPassFilter {
    private _value: Vector3 = Vector3.Zero();
    get value(): Vector3 {
        return this._value;
    }

    constructor(
        private readonly threshold: number
    ) {}

    public update(v: Vector3) {
        if (this.value.subtract(v).length() > this.threshold) {
            this._value = v;
        }
    }

    public reset() {
        this._value = Vector3.Zero();
    }
}

export class FrameMonitor {
    private _lastFrameTimeMs: Nullable<number> = null;

    // Shall only be called once per onResults()
    public sampleFrame(timeMs: number = PrecisionDate.Now) {
        let dt = 0;
        if (this._lastFrameTimeMs !== null) {
            dt = timeMs - this._lastFrameTimeMs;
        }

        this._lastFrameTimeMs = timeMs;
        return dt;
    }
}

export const POSE_LANDMARK_LENGTH = 33;
export const FACE_LANDMARK_LENGTH = 478;
export const HAND_LANDMARK_LENGTH = 21;

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
                    sideOrientation: Mesh.DOUBLESIDE}, this.scene);
        else
            this.arrowInstance = MeshBuilder.ExtrudeShapeCustom(
                "arrow",
                {
                    shape: this.myShape,
                    path: this.myPath,
                    scaleFunction: this.scaling.bind(this),
                    instance: this.arrowInstance}, this.scene);
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
export const cloneableQuaternionToQuaternion = (q: CloneableQuaternion): Quaternion => {
    const ret = new Quaternion(q.x, q.y, q.z, q.w);
    return ret;
};

export function range(start: number, end: number, step: number) {
    return Array.from(
        { length: Math.ceil((end - start) / step) },
        (_, i) => start + i * step
    );
}

export function linspace(start: number, end: number, div: number) {
    const step = (end - start) / div;
    return Array.from(
        {length: div},
        (_, i) => start + i * step
    );
}

export function objectFlip(obj: any) {
    const ret: any = {};
    Object.keys(obj).forEach((key: any) => {
        ret[obj[key]] = key;
    });
    return ret;
}

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

export const rangeCap = (
    v: number,
    min: number,
    max: number
) => {
    if (min > max) {
        const tmp = max;
        max = min;
        min = tmp;
    }
    return Math.max(Math.min(v, max), min);
}
export const remapRange = (
    v: number,
    src_low: number,
    src_high: number,
    dst_low: number,
    dst_high: number
) => {
    return dst_low + (v - src_low) * (dst_high - dst_low) / (src_high - src_low);
};
export const remapRangeWithCap = (
    v: number,
    src_low: number,
    src_high: number,
    dst_low: number,
    dst_high: number
) => {
    const v1 = rangeCap(v, src_low, src_high);
    return dst_low + (v1 - src_low) * (dst_high - dst_low) / (src_high - src_low);
};

export function validVector3(v: Vector3) {
    return Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z);
}
export interface NodeQuaternionMap {
    [name: string] : CloneableQuaternion
}
export interface NodeWorldMatrixMap {
    [name: string] : Matrix
}
export const RadToDeg = (r: number) => {
    return Angle.FromRadians(r).degrees();
}
export const DegToRad = (d: number) => {
    return Angle.FromDegrees(d).radians();
}
export function checkQuaternion(q: Quaternion) {
    return Number.isFinite(q.x) && Number.isFinite(q.y) && Number.isFinite(q.z) && Number.isFinite(q.w);
}
// Same as three.js Quaternion.setFromUnitVectors
export const quaternionBetweenVectors = (
    v1: Vector3, v2: Vector3,
    reverseAngle = false,
    reverseAxis = false,
): Quaternion => {
    // const angle = reverseAngle ? -Math.acos(Vector3.Dot(v1, v2)) : Math.acos(Vector3.Dot(v1, v2));
    const angle = Vector3.GetAngleBetweenVectors(v1, v2, Vector3.Cross(v1, v2))
    const axis = reverseAxis ? Vector3.Cross(v2,v1) : Vector3.Cross(v1,v2);
    axis.normalize();
    return Quaternion.RotationAxis(axis, angle);
};
// From -180 to 180
export const remapDegreeWithCap = (deg: number) => {
    deg = rangeCap(deg, 0, 360);
    return deg < 180 ? deg : deg - 360;
}
export const quaternionToDegrees = (
    q: Quaternion,
    remapDegree=false,
) => {
    const angles = q.toEulerAngles();
    const remapFn = remapDegree ? remapDegreeWithCap : (x: number) => x;
    return new Vector3(
        remapFn(RadToDeg(angles.x)),
        remapFn(RadToDeg(angles.y)),
        remapFn(RadToDeg(angles.z)),
    );
};
export function vectorsSameDirWithinEps(v1: Vector3, v2: Vector3, eps = 1e-6) {
    return v1.cross(v2).length() < eps && Vector3.Dot(v1, v2) > 0;
}
export function testQuaternionEqualsByVector(q1: Quaternion, q2: Quaternion) {
    const testVec = Vector3.One();
    const testVec1 = Vector3.Zero();
    const testVec2 = Vector3.One();
    testVec.rotateByQuaternionToRef(q1, testVec1);
    testVec.rotateByQuaternionToRef(q2, testVec2);
    return vectorsSameDirWithinEps(testVec1, testVec2);
}
export function degreesEqualInQuaternion(
    d1: Vector3, d2: Vector3
) {
    const q1 = Quaternion.FromEulerAngles(DegToRad(d1.x), DegToRad(d1.y), DegToRad(d1.z));
    const q2 = Quaternion.FromEulerAngles(DegToRad(d2.x), DegToRad(d2.y), DegToRad(d2.z));
    return testQuaternionEqualsByVector(q1, q2);
}
export const degreeBetweenVectors = (
    v1: Vector3, v2: Vector3, remapDegree=false
) => {
    return quaternionToDegrees(quaternionBetweenVectors(v1, v2), remapDegree);
};
export enum AXIS {
    x,
    y,
    z,
    xy,
    yz,
    xz,
    xyz,
    none=10
}
export const reverseRotation = (q: Quaternion, axis: AXIS) => {
    if (axis === AXIS.none) return q;
    const angles = q.toEulerAngles();
    switch (axis) {
        case AXIS.x:
            angles.x = -angles.x;
            break;
        case AXIS.y:
            angles.y = -angles.y;
            break;
        case AXIS.z:
            angles.z = -angles.z;
            break;
        case AXIS.xy:
            angles.x = -angles.x;
            angles.y = -angles.y;
            break;
        case AXIS.yz:
            angles.y = -angles.y;
            angles.z = -angles.z;
            break;
        case AXIS.xz:
            angles.x = -angles.x;
            angles.z = -angles.z;
            break;
        case AXIS.xyz:
            angles.x = -angles.x;
            angles.y = -angles.y;
            angles.z = -angles.z;
            break;
        default:
            throw Error("Unknown axis!");
    }
    return Quaternion.RotationYawPitchRoll(angles.y, angles.x, angles.z);
}
// Always remap degrees. Allow capping two axis.
export const removeRotationAxisWithCap = (
    q: Quaternion,
    axis: AXIS,
    capAxis1?: AXIS,
    capLow1?: number,
    capHigh1?: number,
    capAxis2?: AXIS,
    capLow2?: number,
    capHigh2?: number,
) => {
    const angles = quaternionToDegrees(q, true);
    switch (axis) {
        case AXIS.none:
            break;
        case AXIS.x:
            angles.x = 0;
            break;
        case AXIS.y:
            angles.y = 0;
            break;
        case AXIS.z:
            angles.z = 0;
            break;
        case AXIS.xy:
            angles.x = 0;
            angles.y = 0;
            break;
        case AXIS.yz:
            angles.y = 0;
            angles.z = 0;
            break;
        case AXIS.xz:
            angles.x = 0;
            angles.z = 0;
            break;
        case AXIS.xyz:
            angles.x = 0;
            angles.y = 0;
            angles.z = 0;
            break;
        default:
            throw Error("Unknown axis!");
    }
    if (capAxis1 !== undefined && capLow1 !== undefined && capHigh1 !== undefined) {
        switch (capAxis1 as AXIS) {
            case AXIS.x:
                angles.x = rangeCap(angles.x, capLow1, capHigh1);
                break;
            case AXIS.y:
                angles.y = rangeCap(angles.y, capLow1, capHigh1);
                break;
            case AXIS.z:
                angles.z = rangeCap(angles.z, capLow1, capHigh1);
                break;
            default:
                throw Error("Unknown cap axis!");
        }
    }
    if (capAxis2 !== undefined && capLow2 !== undefined && capHigh2 !== undefined) {
        switch (capAxis2 as AXIS) {
            case AXIS.x:
                angles.x = rangeCap(angles.x, capLow2, capHigh2);
                break;
            case AXIS.y:
                angles.y = rangeCap(angles.y, capLow2, capHigh2);
                break;
            case AXIS.z:
                angles.z = rangeCap(angles.z, capLow2, capHigh2);
                break;
            default:
                throw Error("Unknown cap axis!");
        }
    }
    return Quaternion.RotationYawPitchRoll(
        DegToRad(angles.y),
        DegToRad(angles.x),
        DegToRad(angles.z));
}
export const exchangeRotationAxis = (
    q: Quaternion,
    axis1: AXIS,
    axis2: AXIS,
) => {
    const angles: number[] = [];
    q.toEulerAngles().toArray(angles);
    const angle1 = angles[axis1];
    const angle2 = angles[axis2];
    const temp = angle1;
    angles[axis1] = angle2;
    angles[axis2] = temp;
    return Quaternion.FromEulerAngles(
        angles[0],
        angles[1],
        angles[2]);
}

export type KeysMatching<T, V> = { [K in keyof T]-?: T[K] extends V ? K : never }[keyof T];

export function setEqual<T>(as: Set<T>, bs: Set<T>) {
    if (as.size !== bs.size) return false;
    for (const a of as) if (!bs.has(a)) return false;
    return true;
}

// type MethodKeysOfA = KeysMatching<A, Function>;

export type IfEquals<X, Y, A = X, B = never> =
    (<T>() => T extends X ? 1 : 2) extends
        (<T>() => T extends Y ? 1 : 2) ? A : B;
export type ReadonlyKeys<T> = {
    [P in keyof T]-?: IfEquals<{ [Q in P]: T[P] }, { -readonly [Q in P]: T[P] }, never, P>}[keyof T];

// type ReadonlyKeysOfA = ReadonlyKeys<A>;

// Calculate 3D rotations
export type Vector33 = [Vector3, Vector3, Vector3];
export class Basis {
    private static readonly ORIGINAL_CARTESIAN_BASIS_VECTORS: Vector33 = [
        new Vector3(1, 0, 0),
        new Vector3(0, 1, 0),
        new Vector3(0, 0, 1),
    ];

    private readonly _data: Vector33 = Basis.getOriginalCoordVectors();
    get x(): Vector3 {
        return this._data[0];
    }
    get y(): Vector3 {
        return this._data[1];
    }
    get z(): Vector3 {
        return this._data[2];
    }

    constructor(
        v33: Nullable<Vector33>,
        private readonly leftHanded = true,
        private eps=1e-6
    ) {
        if (v33 && v33.every((v) => validVector3(v)))
            this.set(v33);
        this._data.forEach((v) => {
            Object.freeze(v);
        })
    }

    public get() {
        return this._data;
    }

    private set(v33: Vector33) {
        this.x.copyFrom(v33[0]);
        this.y.copyFrom(v33[1]);
        this.z.copyFrom(v33[2]);

        this.verifyBasis();
    }

    public verifyBasis() {
        const z = this.leftHanded ? this.z : this.z.negate();
        if (!vectorsSameDirWithinEps(this.x.cross(this.y), z, this.eps))
            throw Error("Basis is not correct!");
    }

    public rotateByQuaternion(q: Quaternion): Basis {
        const newBasisVectors: Vector33 = [Vector3.Zero(), Vector3.Zero(), Vector3.Zero()];
        this._data.map((v, i) => {
            v.rotateByQuaternionToRef(q, newBasisVectors[i]);
        });
        return new Basis(newBasisVectors);
    }

    // Basis validity is not checked!
    public negateAxes(axis:AXIS) {
        const x = this.x.clone();
        const y = this.y.clone();
        const z = this.z.clone();
        switch (axis) {
            case AXIS.x:
                x.negateInPlace();
                break;
            case AXIS.y:
                y.negateInPlace();
                break;
            case AXIS.z:
                z.negateInPlace();
                break;
            case AXIS.xy:
                x.negateInPlace();
                y.negateInPlace();
                break;
            case AXIS.yz:
                y.negateInPlace();
                z.negateInPlace();
                break;
            case AXIS.xz:
                x.negateInPlace();
                z.negateInPlace();
                break;
            case AXIS.xyz:
                x.negateInPlace();
                y.negateInPlace();
                z.negateInPlace();
                break;
            default:
                throw Error("Unknown axis!");
        }

        return new Basis([x, y, z]);
    }

    public transpose(order: [number, number, number]) {
        // Sanity check
        if (!setEqual<number>(new Set(order), new Set([0, 1, 2]))) {
            console.error("Basis transpose failed: wrong input.");
            return this;
        }

        const data = [this.x.clone(), this.y.clone(), this.z.clone()];
        const newData = order.map(i => data[i]) as Vector33;

        return new Basis(newData);
    }

    private static getOriginalCoordVectors(): Vector33 {
        return Basis.ORIGINAL_CARTESIAN_BASIS_VECTORS.map(v => v.clone()) as Vector33;
    }
}

/*
 * Calculate rotation between two local coordinate systems.
 */
export function quaternionBetweenObj(
    obj1: Vector33,
    obj2: Vector33
): Quaternion {
    const basis1 = getBasis(obj1);
    const basis2 = getBasis(obj2);

    const quaternion = quaternionBetweenBases(basis1, basis2);
    return quaternion;
}

export function printQuaternion(q: Quaternion, s?: string) {
    console.log(s, vectorToNormalizedLandmark(quaternionToDegrees(q, true)));
}

export function quaternionBetweenBases(
    basis1: Basis,
    basis2: Basis,
    extraQuaternion? : Quaternion
) {
    let thisBasis1 = basis1, thisBasis2 = basis2;
    if (extraQuaternion !== undefined) {
        const extraQuaternionR = Quaternion.Inverse(extraQuaternion);
        thisBasis1 = basis1.rotateByQuaternion(extraQuaternionR);
        thisBasis2 = basis2.rotateByQuaternion(extraQuaternionR);
    }
    const rotationBasis1 = Quaternion.RotationQuaternionFromAxis(
        thisBasis1.x.clone(),
        thisBasis1.y.clone(),
        thisBasis1.z.clone());
    const rotationBasis2 = Quaternion.RotationQuaternionFromAxis(
        thisBasis2.x.clone(),
        thisBasis2.y.clone(),
        thisBasis2.z.clone());

    const quaternion31 = rotationBasis1.clone().normalize();
    const quaternion31R = Quaternion.Inverse(quaternion31);
    const quaternion32 = rotationBasis2.clone().normalize();
    const quaternion3 = quaternion32.multiply(quaternion31R);
    return quaternion3;
}

export function test_quaternionBetweenBases3() {
    console.log("Testing quaternionBetweenBases3");

    const remap = false;
    const basis0: Basis = new Basis(null);

    // X 90
    const deg10 = new Vector3(90, 0, 0);
    const basis1 = basis0.rotateByQuaternion(Quaternion.FromEulerAngles(DegToRad(deg10.x), DegToRad(deg10.y), DegToRad(deg10.z)));
    basis1.verifyBasis();
    const degrees1 = quaternionToDegrees(quaternionBetweenBases(basis0, basis1), remap);
    console.log(vectorToNormalizedLandmark(degrees1), degreesEqualInQuaternion(deg10, degrees1));

    // Y 225
    const deg20 = new Vector3(0, 225, 0);
    const basis2 = basis0.rotateByQuaternion(
        Quaternion.FromEulerAngles(DegToRad(deg20.x), DegToRad(deg20.y), DegToRad(deg20.z)));
    basis2.verifyBasis();
    const degrees2 = quaternionToDegrees(quaternionBetweenBases(basis0, basis2), remap);
    console.log(vectorToNormalizedLandmark(degrees2), degreesEqualInQuaternion(deg20, degrees2));

    // Z 135
    const deg30 = new Vector3(0, 0, 135);
    const basis3 = basis0.rotateByQuaternion(
        Quaternion.FromEulerAngles(DegToRad(deg30.x), DegToRad(deg30.y), DegToRad(deg30.z)));
    basis3.verifyBasis();
    const degrees3 = quaternionToDegrees(quaternionBetweenBases(basis0, basis3), remap);
    console.log(vectorToNormalizedLandmark(degrees3), degreesEqualInQuaternion(deg30, degrees3));

    // X Y 135
    const deg40 = new Vector3(135, 135, 0);
    const basis4 = basis0.rotateByQuaternion(
        Quaternion.FromEulerAngles(DegToRad(deg40.x), DegToRad(deg40.y), DegToRad(deg40.z)));
    basis4.verifyBasis();
    const degrees4 = quaternionToDegrees(quaternionBetweenBases(basis0, basis4), remap);
    console.log(vectorToNormalizedLandmark(degrees4), degreesEqualInQuaternion(deg40, degrees4));

    // X Z 90
    const deg50 = new Vector3(90, 0, 90);
    const basis5 = basis0.rotateByQuaternion(
        Quaternion.FromEulerAngles(DegToRad(deg50.x), DegToRad(deg50.y), DegToRad(deg50.z)));
    basis5.verifyBasis();
    const degrees5 = quaternionToDegrees(quaternionBetweenBases(basis0, basis5), remap);
    console.log(vectorToNormalizedLandmark(degrees5), degreesEqualInQuaternion(deg50, degrees5));

    // Y Z 225
    const deg60 = new Vector3(0, 225, 225);
    const basis6 = basis0.rotateByQuaternion(
        Quaternion.FromEulerAngles(DegToRad(deg60.x), DegToRad(deg60.y), DegToRad(deg60.z)));
    basis6.verifyBasis();
    const degrees6 = quaternionToDegrees(quaternionBetweenBases(basis0, basis6), remap);
    console.log(vectorToNormalizedLandmark(degrees6), degreesEqualInQuaternion(deg60, degrees6));

    // X Y Z 135
    const deg70 = new Vector3(135, 135, 135);
    const basis7 = basis0.rotateByQuaternion(
        Quaternion.FromEulerAngles(DegToRad(deg70.x), DegToRad(deg70.y), DegToRad(deg70.z)));
    basis7.verifyBasis();
    const degrees7 = quaternionToDegrees(quaternionBetweenBases(basis0, basis7), remap);
    console.log(vectorToNormalizedLandmark(degrees7), degreesEqualInQuaternion(deg70, degrees7));
}

export function test_getBasis() {
    console.log("Testing getBasis");

    const axes0: Vector33 = [
        new Vector3(0, 0, 0),
        new Vector3(2, 0, 0),
        new Vector3(1, 1, 0)
    ];
    const axes = getBasis(axes0);
    console.log(vectorToNormalizedLandmark(axes.x));
    console.log(vectorToNormalizedLandmark(axes.y));
    console.log(vectorToNormalizedLandmark(axes.z));
}

export function test_quaternionBetweenVectors() {
    console.log("Testing quaternionBetweenVectors");

    const remap = true;
    const vec0 = Vector3.One();

    // X 90
    const vec1 = Vector3.Zero();
    const deg10 = new Vector3(90, 0, 0);
    vec0.rotateByQuaternionToRef(Quaternion.FromEulerAngles(
        DegToRad(deg10.x), DegToRad(deg10.y), DegToRad(deg10.z),
    ), vec1);
    const deg11 = quaternionToDegrees(quaternionBetweenVectors(vec0, vec1), remap);
    console.log(vectorToNormalizedLandmark(deg11), degreesEqualInQuaternion(deg10, deg11));

    // Y 225
    const vec2 = Vector3.Zero();
    const deg20 = new Vector3(0, 225, 0);
    vec0.rotateByQuaternionToRef(Quaternion.FromEulerAngles(
        DegToRad(deg20.x), DegToRad(deg20.y), DegToRad(deg20.z),
    ), vec2);
    const deg21 = quaternionToDegrees(quaternionBetweenVectors(vec0, vec2), remap);
    console.log(vectorToNormalizedLandmark(deg21), degreesEqualInQuaternion(deg20, deg21));

    // Z 135
    const vec3 = Vector3.Zero();
    const deg30 = new Vector3(0, 0, 135);
    vec0.rotateByQuaternionToRef(Quaternion.FromEulerAngles(
        DegToRad(deg30.x), DegToRad(deg30.y), DegToRad(deg30.z),
    ), vec3);
    const deg31 = quaternionToDegrees(quaternionBetweenVectors(vec0, vec3), remap);
    console.log(vectorToNormalizedLandmark(deg31), degreesEqualInQuaternion(deg30, deg31));

    // X Y 90
    const vec4 = Vector3.Zero();
    const deg40 = new Vector3(90, 90, 0);
    vec0.rotateByQuaternionToRef(Quaternion.FromEulerAngles(
        DegToRad(deg40.x), DegToRad(deg40.y), DegToRad(deg40.z),
    ), vec4);
    const deg41 = quaternionToDegrees(quaternionBetweenVectors(vec0, vec4), remap);
    console.log(vectorToNormalizedLandmark(deg41), degreesEqualInQuaternion(deg40, deg41));

    // X Z 135
    const vec5 = Vector3.Zero();
    const deg50 = new Vector3(135, 0, 135);
    vec0.rotateByQuaternionToRef(Quaternion.FromEulerAngles(
        DegToRad(deg50.x), DegToRad(deg50.y), DegToRad(deg50.z),
    ), vec5);
    const deg51 = quaternionToDegrees(quaternionBetweenVectors(vec0, vec5), remap);
    console.log(vectorToNormalizedLandmark(deg51), degreesEqualInQuaternion(deg50, deg51));

    // Y Z 45
    const vec6 = Vector3.Zero();
    const deg60 = new Vector3(0, 45, 45);
    vec0.rotateByQuaternionToRef(Quaternion.FromEulerAngles(
        DegToRad(deg60.x), DegToRad(deg60.y), DegToRad(deg60.z),
    ), vec6);
    const deg61 = quaternionToDegrees(quaternionBetweenVectors(vec0, vec6), remap);
    console.log(vectorToNormalizedLandmark(deg61), degreesEqualInQuaternion(deg60, deg61));

    // X Y Z 135
    const vec7 = Vector3.Zero();
    const deg70 = new Vector3(135, 135, 135);
    vec0.rotateByQuaternionToRef(Quaternion.FromEulerAngles(
        DegToRad(deg70.x), DegToRad(deg70.y), DegToRad(deg70.z),
    ), vec7);
    const deg71 = quaternionToDegrees(quaternionBetweenVectors(vec0, vec7), remap);
    console.log(vectorToNormalizedLandmark(deg71), degreesEqualInQuaternion(deg70, deg71));
}

export function test_calcSphericalCoord(rotationVector = Vector3.Zero()) {
    console.log("Testing calcSphericalCoord");

    const vec0 = new Vector3(1, 1, 1);
    const basisOriginal = new Basis(null);
    const basis0 = basisOriginal.rotateByQuaternion(Quaternion.FromEulerVector(rotationVector));

    // X 90
    const deg10 = new Vector3(90, 0, 0);
    const q11 = Quaternion.FromEulerAngles(
        DegToRad(deg10.x), DegToRad(deg10.y), DegToRad(deg10.z),
    );
    const vec10 = Vector3.Zero();
    vec0.rotateByQuaternionToRef(q11, vec10);
    const [rady1, radz1] = calcSphericalCoord(vec10, basis0, true);
    const vec11 = Vector3.Zero();
    const q12 = sphericalToQuaternion(basis0, rady1, radz1);
    basis0.x.rotateByQuaternionToRef(q12, vec11);
    console.log(RadToDeg(rady1), RadToDeg(radz1), vectorsSameDirWithinEps(vec10, vec11));

    // Y 225
    const deg20 = new Vector3(0, 225, 0);
    const q21 = Quaternion.FromEulerAngles(
        DegToRad(deg20.x), DegToRad(deg20.y), DegToRad(deg20.z),
    );
    const vec20 = Vector3.Zero();
    vec0.rotateByQuaternionToRef(q21, vec20);
    const [rady2, radz2] = calcSphericalCoord(vec20, basis0, true);
    const vec21 = Vector3.Zero();
    const q22 = sphericalToQuaternion(basis0, rady2, radz2);
    basis0.x.rotateByQuaternionToRef(q22, vec21);
    console.log(RadToDeg(rady2), RadToDeg(radz2), vectorsSameDirWithinEps(vec20, vec21));

    // Z 135
    const deg30 = new Vector3(0, 0, 135);
    const q31 = Quaternion.FromEulerAngles(
        DegToRad(deg30.x), DegToRad(deg30.y), DegToRad(deg30.z),
    );
    const vec30 = Vector3.Zero();
    vec0.rotateByQuaternionToRef(q31, vec30);
    const [rady3, radz3] = calcSphericalCoord(vec30, basis0, true);
    const vec31 = Vector3.Zero();
    const q32 = sphericalToQuaternion(basis0, rady3, radz3);
    basis0.x.rotateByQuaternionToRef(q32, vec31);
    console.log(RadToDeg(rady3), RadToDeg(radz3), vectorsSameDirWithinEps(vec30, vec31));

    // X Y 90
    const deg40 = new Vector3(90, 90, 0);
    const q41 = Quaternion.FromEulerAngles(
        DegToRad(deg40.x), DegToRad(deg40.y), DegToRad(deg40.z),
    );
    const vec40 = Vector3.Zero();
    vec0.rotateByQuaternionToRef(q41, vec40);
    const [rady4, radz4] = calcSphericalCoord(vec40, basis0, true);
    const vec41 = Vector3.Zero();
    const q42 = sphericalToQuaternion(basis0, rady4, radz4);
    basis0.x.rotateByQuaternionToRef(q42, vec41);
    console.log(RadToDeg(rady4), RadToDeg(radz4), vectorsSameDirWithinEps(vec40, vec41));

    // X Z 135
    const deg50 = new Vector3(135, 0, 135);
    const q51 = Quaternion.FromEulerAngles(
        DegToRad(deg50.x), DegToRad(deg50.y), DegToRad(deg50.z),
    );
    const vec50 = Vector3.Zero();
    vec0.rotateByQuaternionToRef(q51, vec50);
    const [rady5, radz5] = calcSphericalCoord(vec50, basis0, true);
    const vec51 = Vector3.Zero();
    const q52 = sphericalToQuaternion(basis0, rady5, radz5);
    basis0.x.rotateByQuaternionToRef(q52, vec51);
    console.log(RadToDeg(rady5), RadToDeg(radz5), vectorsSameDirWithinEps(vec50, vec51));

    // Y Z 45
    const deg60 = new Vector3(0, 45, 45);
    const q61 = Quaternion.FromEulerAngles(
        DegToRad(deg60.x), DegToRad(deg60.y), DegToRad(deg60.z),
    );
    const vec60 = Vector3.Zero();
    vec0.rotateByQuaternionToRef(q61, vec60);
    const [rady6, radz6] = calcSphericalCoord(vec60, basis0, true);
    const vec61 = Vector3.Zero();
    const q62 = sphericalToQuaternion(basis0, rady6, radz6);
    basis0.x.rotateByQuaternionToRef(q62, vec61);
    console.log(RadToDeg(rady6), RadToDeg(radz6), vectorsSameDirWithinEps(vec60, vec61));

    // X Y Z 135
    const deg70 = new Vector3(135, 135, 135);
    const q71 = Quaternion.FromEulerAngles(
        DegToRad(deg70.x), DegToRad(deg70.y), DegToRad(deg70.z),
    );
    const vec70 = Vector3.Zero();
    vec0.rotateByQuaternionToRef(q71, vec70);
    const [rady7, radz7] = calcSphericalCoord(vec70, basis0, true);
    const vec71 = Vector3.Zero();
    const q72 = sphericalToQuaternion(basis0, rady7, radz7);
    basis0.x.rotateByQuaternionToRef(q72, vec71);
    console.log(RadToDeg(rady7), RadToDeg(radz7), vectorsSameDirWithinEps(vec70, vec71));
}

/*
 * Left handed for BJS.
 * Each object is defined by 3 points.
 * Assume a is origin, b points to +x, abc forms XY plane.
 */
export function getBasis(obj: Vector33): Basis {
    const [a, b, c] = obj;
    const planeXY = Plane.FromPoints(a, b, c).normalize();
    const axisX = b.subtract(a).normalize();
    const axisZ = planeXY.normal;
    // Project c onto ab
    const cp = a.add(
        axisX.scale(Vector3.Dot(c.subtract(a), axisX) / Vector3.Dot(axisX, axisX))
    );
    const axisY = c.subtract(cp).normalize();
    return new Basis([axisX, axisY, axisZ]);
}
// Project points to an average plane
export function calcAvgPlane(pts: Vector3[], normal: Vector3): Vector3[] {
    if (pts.length === 0) return [Vector3.Zero()];
    const avgPt = pts.reduce((prev, curr) => {
        return prev.add(curr);
    }).scale(1 / pts.length);

    const ret = pts.map((v) => {
        return v.subtract(normal.scale(Vector3.Dot(normal, v.subtract(avgPt))))
    });

    return ret;
}

// Result is in Radian on unit sphere (r = 1).
export function calcSphericalCoord0(
    pos: Vector3, basis: Basis,
) {
    const qToOriginal = Quaternion.Inverse(Quaternion.RotationQuaternionFromAxis(
        basis.x.clone(), basis.y.clone(), basis.z.clone())).normalize();
    const posInOriginal = Vector3.Zero();
    pos.rotateByQuaternionToRef(qToOriginal, posInOriginal);
    posInOriginal.normalize();

    // Calculate theta and phi
    const x = posInOriginal.x;
    let y = posInOriginal.y;
    let z = posInOriginal.z;

    const theta = Math.acos(z);
    const phi = Math.atan2(y, x);

    return [theta, phi];
}
// Modified version for fingers, only allow -90 < phi < 90, but -180 < theta < 180.
export function calcSphericalCoord(
    pos: Vector3, basis: Basis,
    testMode = false) {
    const qToOriginal = Quaternion.Inverse(quaternionBetweenBases(
        new Basis(null), basis));
    const posInOriginal = Vector3.Zero();
    pos.rotateByQuaternionToRef(qToOriginal, posInOriginal);
    posInOriginal.normalize();

    // Calculate theta and phi
    const x = posInOriginal.x;
    const y = posInOriginal.y;
    const z = posInOriginal.z;

    let theta, phi;
    if (x != 0) {
        theta = Math.sign(x) * Math.acos(z);
        phi = Math.atan(y / x);
    } else {
        if (y != 0) {
            theta = Math.sign(y) * Math.acos(z);
            phi = Math.PI / 2;
        } else {
            theta = Math.acos(z);
            phi = 0;
        }
    }

    // Special case for irregular landmakrs
    if (theta < (-Math.PI / 6)) {
        theta = Math.PI / 2 - theta;
    }

    return [theta, phi];
}
// Assuming rotation starts from (1, 0, 0) in given coordinate system.
export function sphericalToQuaternion(basis: Basis, theta: number, phi: number) {
    const xTz = Quaternion.RotationAxis(basis.y.clone(), -Math.PI / 2);
    const q1 = Quaternion.RotationAxis(basis.z.clone(), phi);
    const y1 = Vector3.Zero();
    basis.y.rotateByQuaternionToRef(q1, y1);
    const q2 = Quaternion.RotationAxis(y1, theta);
    return q2.multiply(q1.multiply(xTz));

}

export function projectVectorOnPlane(projPlane: Plane, vec: Vector3) {
    return vec.subtract(projPlane.normal.scale(Vector3.Dot(vec, projPlane.normal)));
}
export function round(value: number, precision: number) {
    const multiplier = Math.pow(10, precision || 0);
    return Math.round(value * multiplier) / multiplier;
}
