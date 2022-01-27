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

import {Nullable, Quaternion, Angle, Matrix, Vector3, Curve3, Plane} from "@babylonjs/core";
import {rangeCap} from "./utils";
import {Basis, quaternionBetweenBases} from "./basis";
import {vectorToNormalizedLandmark} from "./landmark";
import {
    FilterParams,
    GaussianVectorFilter,
    KalmanVectorFilter,
} from "./filter";

export class CloneableQuaternionLite {
    public x: number = 0;
    public y: number = 0;
    public z: number = 0;
    public w: number = 1;

    constructor(
        q: Nullable<Quaternion>,
    ) {
        if (q) {
            this.x = q.x;
            this.y = q.y;
            this.z = q.z;
            this.w = q.w;
        }
    }
}
export class CloneableQuaternion extends CloneableQuaternionLite {
    private readonly _baseBasis: Basis;
    get baseBasis(): Basis {
        return this._baseBasis;
    }

    constructor(
        q: Nullable<Quaternion>,
        basis?: Basis
    ) {
        super(q);
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

/**
 * Catmull-Rom Spline interpolated rotation angles.
 * Handles unevenly timed frames.
 */
export class CloneableRotationAngleCurve3 {
    private _rotationQuaternions: CloneableQuaternionLite[] = [];
    private _pointsX: Vector3[] = [];
    get pointsX(): Vector3[] {
        return this._pointsX;
    }
    private _pointsY: Vector3[] = [];
    get pointsY(): Vector3[] {
        return this._pointsY;
    }
    private _pointsZ: Vector3[] = [];
    get pointsZ(): Vector3[] {
        return this._pointsZ;
    }

    constructor(public curveLength: number) {
        if (!Number.isSafeInteger(curveLength)) {
            console.warn("Curve length is not an integer!");
            this.curveLength = Math.floor(curveLength);
        }
    }

    public createCurvePoints(
        qs: Vector3[],
        ts: number[],
    ) {
        // Sanity check
        if (qs.length !== ts.length || ts.length < 4 ||!ts.every((v, i, self) => {
            if (i < 1) return true;
            else return v > self[i - 1];
        })) return;

        this._pointsX = Curve3.CreateCatmullRomSpline(
            qs.map((v, i) => new Vector3(ts[i], v.x, 0)),
            this.curveLength, false).getPoints();
        this._pointsY = Curve3.CreateCatmullRomSpline(
            qs.map((v, i) => new Vector3(ts[i], v.y, 0)),
            this.curveLength, false).getPoints();
        this._pointsZ = Curve3.CreateCatmullRomSpline(
            qs.map((v, i) => new Vector3(ts[i], v.z, 0)),
            this.curveLength, false).getPoints();
    }
}

export interface CloneableQuaternionMap {
    [key: string]: CloneableQuaternion
}
export type CloneableQuaternionList = CloneableQuaternion[];
export const cloneableQuaternionToQuaternion = (q: CloneableQuaternionLite): Quaternion => {
    const ret = new Quaternion(q.x, q.y, q.z, q.w);
    return ret;
};

export class FilteredQuaternion {
    private mainFilter: KalmanVectorFilter;
    private gaussianVectorFilter: Nullable<GaussianVectorFilter> = null;

    private _t = 0;
    get t(): number {
        return this._t;
    }
    set t(value: number) {
        this._t = value;
    }

    private _rot = Quaternion.Identity();
    get rot(): Quaternion {
        return this._rot;
    }

    constructor(
        params: FilterParams = {
            R: 1,
            Q: 1,
            type: 'Kalman'
        }
    ) {
        if (params.type === "Kalman")
            this.mainFilter = new KalmanVectorFilter(params.R, params.Q);
        else
            throw Error("Wrong filter type!");
        if (params.gaussianSigma)
            this.gaussianVectorFilter = new GaussianVectorFilter(5, params.gaussianSigma);
    }

    public updateRotation(rot: Quaternion) {
        this.t += 1;
        let angles = rot.toEulerAngles();
        angles = this.mainFilter.next(this.t, angles);

        if (this.gaussianVectorFilter) {
            this.gaussianVectorFilter.push(angles);
            angles = this.gaussianVectorFilter.apply();
        }

        this._rot = Quaternion.FromEulerVector(angles);
    }
}

export type FilteredQuaternionList = FilteredQuaternion[];


export interface NodeWorldMatrixMap {
    [name: string] : Matrix
}
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

// Convenience functions
export const RadToDeg = (r: number) => {
    return Angle.FromRadians(r).degrees();
}
export const DegToRad = (d: number) => {
    return Angle.FromDegrees(d).radians();
}

/**
 * Check a quaternion is valid
 * @param q Input quaternion
 */
export function checkQuaternion(q: Quaternion) {
    return Number.isFinite(q.x) && Number.isFinite(q.y) && Number.isFinite(q.z) && Number.isFinite(q.w);
}
// Similar to three.js Quaternion.setFromUnitVectors
export const quaternionBetweenVectors = (
    v1: Vector3, v2: Vector3,
): Quaternion => {
    const angle = Vector3.GetAngleBetweenVectors(v1, v2, Vector3.Cross(v1, v2))
    const axis = Vector3.Cross(v1,v2);
    axis.normalize();
    return Quaternion.RotationAxis(axis, angle);
};
/**
 * Same as above, Euler angle version
 * @param v1 Input rotation in degrees 1
 * @param v2 Input rotation in degrees 2
 * @param remapDegree Whether re-map degrees
 */
export const degreeBetweenVectors = (
    v1: Vector3, v2: Vector3, remapDegree=false
) => {
    return quaternionToDegrees(quaternionBetweenVectors(v1, v2), remapDegree);
};
/**
 * Re-map degrees to -180 to 180
 * @param deg Input angle in Degrees
 */
export const remapDegreeWithCap = (deg: number) => {
    deg = rangeCap(deg, 0, 360);
    return deg < 180 ? deg : deg - 360;
}
/**
 * Convert quaternions to degrees
 * @param q Input quaternion
 * @param remapDegree whether re-map degrees
 */
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

/**
 * Check whether two directions are close enough within a small values
 * @param v1 Input direction 1
 * @param v2 Input direction 2
 * @param eps Error threshold
 */
export function vectorsSameDirWithinEps(v1: Vector3, v2: Vector3, eps = 1e-6) {
    return v1.cross(v2).length() < eps && Vector3.Dot(v1, v2) > 0;
}

/**
 * Test whether two quaternions have equal effects
 * @param q1 Input quaternion 1
 * @param q2 Input quaternion 2
 */
export function testQuaternionEqualsByVector(q1: Quaternion, q2: Quaternion) {
    const testVec = Vector3.One();
    const testVec1 = Vector3.Zero();
    const testVec2 = Vector3.One();
    testVec.rotateByQuaternionToRef(q1, testVec1);
    testVec.rotateByQuaternionToRef(q2, testVec2);
    return vectorsSameDirWithinEps(testVec1, testVec2);
}

/**
 * Same as above, Euler angle version
 * @param d1 Input degrees 1
 * @param d2 Input degrees 2
 */
export function degreesEqualInQuaternion(
    d1: Vector3, d2: Vector3
) {
    const q1 = Quaternion.FromEulerAngles(DegToRad(d1.x), DegToRad(d1.y), DegToRad(d1.z));
    const q2 = Quaternion.FromEulerAngles(DegToRad(d2.x), DegToRad(d2.y), DegToRad(d2.z));
    return testQuaternionEqualsByVector(q1, q2);
}

/**
 * Reverse rotation Euler angles on given axes
 * @param q Input quaternion
 * @param axis Axes to reverse
 */
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
/**
 * Remove rotation on given axes.
 * Optionally capping rotation (in Euler angles) on two axes.
 * This operation assumes re-mapped degrees.
 * @param q Input quaternion
 * @param axis Axes to remove
 * @param capAxis1 Capping axis 1
 * @param capLow1 Axis 1 lower range
 * @param capHigh1 Axis 1 higher range
 * @param capAxis2 Capping axis 2
 * @param capLow2 Axis 2 lower range
 * @param capHigh2 Axis 2 higher range
 */
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
/**
 * Switch rotation axes.
 * @param q Input quaternion
 * @param axis1 Axis 1 to switch
 * @param axis2 Axis 2 to switch
 */
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


/**
 * Result is in Radian on unit sphere (r = 1).
 * Canonical ISO 80000-2:2019 convention.
 * @param pos Euclidean local position
 * @param basis Local coordinate system basis
 */
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
    const y = posInOriginal.y;
    const z = posInOriginal.z;

    const theta = Math.acos(z);
    const phi = Math.atan2(y, x);

    return [theta, phi];
}

/**
 * Modified version for fingers.
 * Outputs -90deg <= phi <= 90deg, -180deg <= theta <= 180deg.
 * @param pos Euclidean local position
 * @param basis Local coordinate system basis
 */
export function calcSphericalCoord(
    pos: Vector3, basis: Basis, isFinger=true
) {
    const qToOriginal = Quaternion.Inverse(Quaternion.RotationQuaternionFromAxis(
        basis.x.clone(), basis.y.clone(), basis.z.clone())).normalize();
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
    if (isFinger && theta < (-Math.PI / 6)) {
        theta = Math.PI / 2 - theta;
    }

    return [theta, phi];
}
/**
 * Assuming rotation starts from (1, 0, 0) in local coordinate system.
 * @param basis Local coordinate system basis
 * @param theta Polar angle
 * @param phi Azimuthal angle
 */
export function sphericalToQuaternion(basis: Basis, theta: number, phi: number) {
    const xTz = Quaternion.RotationAxis(basis.y.clone(), -Math.PI / 2);
    const q1 = Quaternion.RotationAxis(basis.x.clone(), phi);

    const q2 = Quaternion.RotationAxis(basis.y.clone(), theta);

    // Force result to face front
    const planeXZ = Plane.FromPositionAndNormal(Vector3.Zero(), basis.y.clone());
    const intermBasis = basis.rotateByQuaternion(xTz.multiply(q1).multiplyInPlace(q2));
    const newBasisZ = Vector3.Cross(intermBasis.x.clone(), planeXZ.normal);
    const newBasisY = Vector3.Cross(newBasisZ, intermBasis.x.clone());
    const newBasis = new Basis([intermBasis.x, newBasisY, newBasisZ]);

    return quaternionBetweenBases(basis, newBasis);
}

// Scale rotation angles in place
export function scaleRotationInPlace(quaternion: Quaternion, scale: number) {
    const angles = quaternion.toEulerAngles();
    angles.scaleInPlace(scale);
    return Quaternion.FromEulerVectorToRef(angles, quaternion);
}
