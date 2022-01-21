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

import {Basis, getBasis, quaternionBetweenBases, Vector33} from "./basis";
import {Quaternion, Vector3} from "@babylonjs/core";
import {
    calcSphericalCoord,
    degreesEqualInQuaternion,
    DegToRad,
    quaternionBetweenVectors,
    quaternionToDegrees, RadToDeg, sphericalToQuaternion, vectorsSameDirWithinEps
} from "./quaternion";
import {vectorToNormalizedLandmark} from "./landmark";

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
