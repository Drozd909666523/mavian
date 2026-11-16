//! Vectors and 3×3 matrices.
//!
//! Axes: Y up, a character at rest faces +Z, its left side is +X.
//! Additions happen in the same order as in the prototype: the last bits
//! depend on it, and so does frame-for-frame agreement.

use crate::m;
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub const fn v3(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3 { x, y, z }
}

impl Vec3 {
    pub const ZERO: Vec3 = v3(0.0, 0.0, 0.0);
    pub const UP: Vec3 = v3(0.0, 1.0, 0.0);

    pub fn dot(self, b: Vec3) -> f64 {
        self.x * b.x + self.y * b.y + self.z * b.z
    }

    pub fn cross(self, b: Vec3) -> Vec3 {
        v3(
            self.y * b.z - self.z * b.y,
            self.z * b.x - self.x * b.z,
            self.x * b.y - self.y * b.x,
        )
    }

    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    /// Unit vector; a zero vector stays zero.
    pub fn norm(self) -> Vec3 {
        let n = self.length();
        if n < 1e-12 { Vec3::ZERO } else { v3(self.x / n, self.y / n, self.z / n) }
    }

    pub fn lerp(self, b: Vec3, t: f64) -> Vec3 {
        v3(mix(self.x, b.x, t), mix(self.y, b.y, t), mix(self.z, b.z, t))
    }

    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

impl From<(f64, f64, f64)> for Vec3 {
    fn from((x, y, z): (f64, f64, f64)) -> Self {
        v3(x, y, z)
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, b: Vec3) -> Vec3 {
        v3(self.x + b.x, self.y + b.y, self.z + b.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, b: Vec3) -> Vec3 {
        v3(self.x - b.x, self.y - b.y, self.z - b.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f64) -> Vec3 {
        v3(self.x * s, self.y * s, self.z * s)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        v3(-self.x, -self.y, -self.z)
    }
}

/// Linear interpolation of numbers.
pub fn mix(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// 3×3 matrix, row-major.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3(pub [[f64; 3]; 3]);

impl Mat3 {
    pub const IDENT: Mat3 = Mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

    /// Rotation about X: the Y axis turns towards Z.
    pub fn rot_x(deg: f64) -> Mat3 {
        let (s, c) = m::sin_cos(deg.to_radians());
        Mat3([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    }

    /// Rotation about Y: the Z axis turns towards X.
    pub fn rot_y(deg: f64) -> Mat3 {
        let (s, c) = m::sin_cos(deg.to_radians());
        Mat3([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    }

    /// Rotation about Z: the X axis turns towards Y.
    pub fn rot_z(deg: f64) -> Mat3 {
        let (s, c) = m::sin_cos(deg.to_radians());
        Mat3([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    }

    /// Scale by `along` along the unit vector `d` and by `across` across it.
    pub fn scale_along(d: Vec3, along: f64, across: f64) -> Mat3 {
        let k = along - across;
        let d = d.to_array();
        let mut m = [[0.0; 3]; 3];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (if i == j { across } else { 0.0 }) + k * d[i] * d[j];
            }
        }
        Mat3(m)
    }

    /// Column: where local axis i points.
    pub fn col(&self, i: usize) -> Vec3 {
        v3(self.0[0][i], self.0[1][i], self.0[2][i])
    }

    pub fn row(&self, i: usize) -> Vec3 {
        v3(self.0[i][0], self.0[i][1], self.0[i][2])
    }

    /// The inverse of a rotation.
    pub fn transpose(&self) -> Mat3 {
        let m = &self.0;
        Mat3([[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]])
    }

    /// Heading of a rotation about the vertical, degrees (where its Z axis looks).
    pub fn heading(&self) -> f64 {
        let z = self.col(2);
        m::atan2(z.x, z.z).to_degrees()
    }
}

impl Mul for Mat3 {
    type Output = Mat3;
    fn mul(self, b: Mat3) -> Mat3 {
        let (a, b) = (self.0, b.0);
        let mut m = [[0.0; 3]; 3];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        Mat3(m)
    }
}

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        v3(self.row(0).dot(v), self.row(1).dot(v), self.row(2).dot(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotations_follow_axis_rules() {
        let z = Mat3::rot_y(90.0) * v3(0.0, 0.0, 1.0);
        assert!((z - v3(1.0, 0.0, 0.0)).length() < 1e-12, "Z turns towards X");
        let y = Mat3::rot_x(90.0) * v3(0.0, 1.0, 0.0);
        assert!((y - v3(0.0, 0.0, 1.0)).length() < 1e-12, "Y turns towards Z");
    }

    #[test]
    fn scale_along_keeps_cross_axes() {
        let m = Mat3::scale_along(Vec3::UP, 0.5, 2.0);
        assert_eq!(m * v3(0.0, 1.0, 0.0), v3(0.0, 0.5, 0.0));
        assert_eq!(m * v3(1.0, 0.0, 0.0), v3(2.0, 0.0, 0.0));
    }
}
