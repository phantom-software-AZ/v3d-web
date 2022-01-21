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

import {Nullable, Quaternion, Angle, Matrix, Vector3} from "@babylonjs/core";
import {rangeCap} from "./utils";
import {Basis, quaternionBetweenBases} from "./basis";
import {vectorToNormalizedLandmark} from "./landmark";

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
export const cloneableQuaternionToQuaternion = (q: CloneableQuaternion): Quaternion => {
    const ret = new Quaternion(q.x, q.y, q.z, q.w);
    return ret;
};

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

export function printQuaternion(q: Quaternion, s?: string) {
    console.log(s, vectorToNormalizedLandmark(quaternionToDegrees(q, true)));
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
