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

// Calculate 3D rotations
import {Nullable, Plane, Quaternion, Vector3} from "@babylonjs/core";
import {AXIS, vectorsSameDirWithinEps} from "./quaternion";
import {setEqual, validVector3} from "./utils";

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
        private eps = 1e-6
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
    public negateAxes(axis: AXIS) {
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

export function quaternionBetweenBases(
    basis1: Basis,
    basis2: Basis,
    prevQuaternion?: Quaternion
) {
    let thisBasis1 = basis1, thisBasis2 = basis2;
    if (prevQuaternion !== undefined) {
        const extraQuaternionR = Quaternion.Inverse(prevQuaternion);
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
    return quaternion32.multiply(quaternion31R);
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
