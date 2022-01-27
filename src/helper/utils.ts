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
    Nullable,
    Plane,
    PrecisionDate,
    Quaternion, Vector2,
    Vector3
} from "@babylonjs/core";
import {
    calcSphericalCoord, degreesEqualInQuaternion,
    DegToRad,
    quaternionBetweenVectors,
    quaternionToDegrees,
    RadToDeg,
    sphericalToQuaternion,
    vectorsSameDirWithinEps
} from "./quaternion";
import {Basis, getBasis, quaternionBetweenBases, Vector33} from "./basis";
import {vectorToNormalizedLandmark} from "./landmark";

export function initArray<T>(length: number, initializer: (i: number) => T) {
    let arr = new Array<T>(length);
    for (let i = 0; i < length; i++)
        arr[i] = initializer(i);
    return arr;
}

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

export type KeysMatching<T, V> = { [K in keyof T]-?: T[K] extends V ? K : never }[keyof T];

// type MethodKeysOfA = KeysMatching<A, Function>;

export type IfEquals<X, Y, A = X, B = never> =
    (<T>() => T extends X ? 1 : 2) extends
        (<T>() => T extends Y ? 1 : 2) ? A : B;
export type ReadonlyKeys<T> = {
    [P in keyof T]-?: IfEquals<{ [Q in P]: T[P] }, { -readonly [Q in P]: T[P] }, never, P>}[keyof T];

// type ReadonlyKeysOfA = ReadonlyKeys<A>;

export function setEqual<T>(as: Set<T>, bs: Set<T>) {
    if (as.size !== bs.size) return false;
    for (const a of as) if (!bs.has(a)) return false;
    return true;
}

export function projectVectorOnPlane(projPlane: Plane, vec: Vector3) {
    return vec.subtract(projPlane.normal.scale(Vector3.Dot(vec, projPlane.normal)));
}
export function round(value: number, precision: number) {
    const multiplier = Math.pow(10, precision || 0);
    return Math.round(value * multiplier) / multiplier;
}

/**
 * Simple fixed length FIFO queue.
 */
export class fixedLengthQueue<T> {
    private _values: T[] = [];
    get values(): T[] {
        return this._values;
    }

    constructor(public readonly size: number) {}

    public push(v: T) {
        this.values.push(v);

        if (this.values.length === this.size + 1) {
            this.values.shift();
        } else if (this.values.length > this.size + 1) {
            console.warn(`Internal queue has length longer than size ${this.size}: Got length ${this.values.length}`);
            this._values = this.values.slice(-this.size);
        }
    }

    public concat(arr: T[]) {
        this._values = this.values.concat(arr);

        if (this.values.length > this.size) {
            this._values = this.values.slice(-this.size);
        }
    }

    public pop() {
        return this.values.shift();
    }

    public first() {
        if (this._values.length > 0)
            return this.values[0];
        else
            return null;
    }

    public last() {
        if (this._values.length > 0)
            return this._values[this._values.length - 1];
        else
            return null;
    }

    public reset() {
        this.values.length = 0;
    }

    public length() {
        return this.values.length;
    }
}
export const LR = ["left", "right"];
