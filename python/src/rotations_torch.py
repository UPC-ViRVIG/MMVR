import torch
import torch.nn.functional as F

# Source of some functions: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py


def mul_quat(quaternions1: torch.Tensor, quaternions2: torch.Tensor) -> torch.Tensor:
    """
    Multiplies two quaternions.
    Args:
        quaternions1: (x, y, z, w) as tensor of shape (..., 4).
        quaternions2: (x, y, z, w) as tensor of shape (..., 4).
    Returns:
        quaternions: (x, y, z, w) as tensor of shape (..., 4).
    """
    x1, y1, z1, w1 = torch.unbind(quaternions1, -1)
    x2, y2, z2, w2 = torch.unbind(quaternions2, -1)
    rw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    rx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    ry = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    rz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return standardize_quaternion(
        torch.stack(
            (rx, ry, rz, rw),
            -1,
        )
    )


def mul_mat_vec(matrices: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """
    Multiply a matrix by a vector.
    Args:
        matrices: as tensor of shape (..., 9). Matrix order: (c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z) where ci is column i.
        vectors: as tensor of shape (..., 3).
    Returns:
        vectors: as tensor of shape (..., 3).
    """
    m = matrices.clone().reshape(-1, 3, 3).transpose(-2, -1)
    v = vectors.unsqueeze(-1)
    return torch.matmul(m, v).squeeze(-1)


def mul_rot_mat(rotations1: torch.Tensor, rotations2: torch.Tensor) -> torch.Tensor:
    """
    Multiplies two quaternions.
    Args:
        rotations1: as tensor of shape (..., 9). Matrix order: (c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z) where ci is column i.
        rotations2: as tensor of shape (..., 9). Matrix order: (c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z) where ci is column i.
    Returns:
        rotations: as tensor of shape (..., 9). Matrix order: (c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z) where ci is column i.
    """
    r1 = rotations1.clone().reshape(-1, 3, 3).transpose(-2, -1)
    r2 = rotations2.clone().reshape(-1, 3, 3).transpose(-2, -1)
    r = torch.transpose(torch.matmul(r1, r2), -1, -2)
    return r.reshape(-1, 9)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.
    Args:
        quaternions: (x, y, z, w) as tensor of shape (..., 4).
    Returns:
        Standardized quaternions (x, y, z, w) as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def continuous_to_quat(rotations: torch.Tensor) -> torch.Tensor:
    """
    Convert continuous representations to quaternions.
    Args:
        Continuous representation as tensor of shape (..., 6) Matrix order: (c0.x, c0.y, c0.z, c1.x, c1.y, c1.z) where ci is column i.
    Returns:
        quat: (x, y, z, w) as tensor of shape (..., 4).
    """
    b1 = F.normalize(rotations[..., :3], p=2, dim=-1)
    b2 = F.normalize(
        rotations[..., 3:] - (b1 * rotations[..., 3:]).sum(-1).unsqueeze(-1) * b1,
        p=2,
        dim=-1,
    )
    b3 = torch.cross(b1, b2, dim=-1)
    rotations = torch.cat([b1, b2, b3], dim=-1)
    return matrix3x3_to_quat(rotations)


def continuous_to_mat(rotations: torch.Tensor) -> torch.Tensor:
    """
    Convert continuous representations to matrices.
    Args:
        Continuous representation as tensor of shape (..., 6) Matrix order: (c0.x, c0.y, c0.z, c1.x, c1.y, c1.z) where ci is column i.
    Returns:
        mat: as tensor of shape (..., 9). Matrix order: (c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z) where ci is column i.
    """
    b1 = F.normalize(rotations[..., :3], p=2, dim=-1)
    b2 = F.normalize(
        rotations[..., 3:] - (b1 * rotations[..., 3:]).sum(-1).unsqueeze(-1) * b1,
        p=2,
        dim=-1,
    )
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.cat([b1, b2, b3], dim=-1)


def quat_to_continuous(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to continuous representations.
    Args:
        quat: (x, y, z, w) as tensor of shape (..., 4).
    Returns:
        Continuous representation as tensor of shape (..., 6) Matrix order: (c0.x, c0.y, c0.z, c1.x, c1.y, c1.z) where ci is column i.
    """
    rotations = quat_to_matrix3x3(quaternions)
    return rotations[..., :6]


def mat_to_continuous(rotations: torch.Tensor) -> torch.Tensor:
    """
    Convert matrices to continuous representations.
    Args:
        mat: as tensor of shape (..., 9). Matrix order: (c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z) where ci is column i.
    Returns:
        Continuous representation as tensor of shape (..., 6) Matrix order: (c0.x, c0.y, c0.z, c1.x, c1.y, c1.z) where ci is column i.
    """
    return rotations[..., :6]


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix3x3_to_quat(rotations: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions.
    Args:
        rotations: as tensor of shape (..., 9). Matrix order: (c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z) where ci is column i.
    Returns:
        Quaternions (x, y, z, w) as tensor of shape (..., 4).
    """
    # Separate components
    r00, r10, r20, r01, r11, r21, r02, r12, r22 = torch.unbind(rotations, -1)
    # Quaternion
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + r00 + r11 + r22,
                1.0 + r00 - r11 - r22,
                1.0 - r00 + r11 - r22,
                1.0 - r00 - r11 + r22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, r21 - r12, r02 - r20, r10 - r01], dim=-1),
            torch.stack([r21 - r12, q_abs[..., 1] ** 2, r10 + r01, r02 + r20], dim=-1),
            torch.stack([r02 - r20, r10 + r01, q_abs[..., 2] ** 2, r12 + r21], dim=-1),
            torch.stack([r10 - r01, r20 + r02, r21 + r12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    # Reorder from r, i, j, k to x, y, z, w
    quat_candidates = quat_candidates[..., [1, 2, 3, 0]]

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ]


def quat_to_matrix3x3(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: (x, y, z, w) as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 9) Matrix order: (c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z) where ci is column i.
    """
    # Separate components
    x, y, z, w = torch.unbind(quaternions, -1)
    # 1st Row
    r00 = 2 * (w * w + x * x) - 1
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    # 2nd Row
    r10 = 2 * (x * y + w * z)
    r11 = 2 * (w * w + y * y) - 1
    r12 = 2 * (y * z - w * x)
    # 3rd Row
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 2 * (w * w + z * z) - 1
    # Concatenate
    return torch.stack([r00, r10, r20, r01, r11, r21, r02, r12, r22], -1)


def test_quat_matrix3x3():
    # Test from Unity
    q1 = torch.tensor([0.0, 0.0, 1.0, -4.371139e-08])
    q2 = torch.tensor([0.0, 0.309017, 0.0, 0.9510565])
    q3 = torch.tensor([1.0, 0.0, 0.0, -4.371139e-08])
    q4 = torch.tensor([-1.672763e-08, -0.3826835, 0.9238795, -4.038406e-08])
    q5 = torch.tensor([-0.7071068, -3.090862e-08, -0.7071068, -3.090862e-08])
    q5AfterMatrix = torch.tensor([0.707106, 3.090862e-08, 0.7071068, 3.090862e-08])
    q6 = torch.tensor([-0.8660254, 0.5, -2.185569e-08, -3.785517e-08])
    q6AfterMatrix = torch.tensor([0.8660254, -0.5, -2.185569e-08, -3.785517e-08])
    quaternions = torch.stack([q1, q2, q3, q4, q5, q6], 0)
    quaternionsAfterMatrix = torch.stack(
        [q1, q2, q3, q4, q5AfterMatrix, q6AfterMatrix], 0
    )
    m1 = torch.tensor([-1, -8.742278e-08, 0, 8.742278e-08, -1, 0, 0, 0, 1])
    m2 = torch.tensor([0.809017, 0, -0.5877853, 0, 1, 0, 0.5877853, 0, 0.809017])
    m3 = torch.tensor([1, 0, 0, 0, -1, -8.742278e-08, 0, 8.742278e-08, -1])
    m4 = torch.tensor(
        [
            -1,
            -6.181723e-08,
            -6.181724e-08,
            8.742278e-08,
            -0.7071067,
            -0.7071068,
            0,
            -0.7071068,
            0.7071067,
        ]
    )
    m5 = torch.tensor(
        [
            5.960464e-08,
            8.742278e-08,
            0.9999999,
            0,
            -0.9999999,
            8.742278e-08,
            0.9999999,
            0,
            5.960464e-08,
        ]
    )
    m6 = torch.tensor(
        [
            0.5,
            -0.8660254,
            7.571035e-08,
            -0.8660254,
            -0.5,
            4.371138e-08,
            0,
            -8.742278e-08,
            -1,
        ]
    )
    expected_matrices = torch.stack([m1, m2, m3, m4, m5, m6], 0)
    # quat to matrix3x3
    assert torch.allclose(quat_to_matrix3x3(q1), m1, atol=1e-7)
    assert torch.allclose(quat_to_matrix3x3(q2), m2, atol=1e-7)
    assert torch.allclose(quat_to_matrix3x3(q3), m3, atol=1e-7)
    assert torch.allclose(quat_to_matrix3x3(q4), m4, atol=1e-7)
    assert torch.allclose(quat_to_matrix3x3(q5), m5, atol=1e-7)
    assert torch.allclose(quat_to_matrix3x3(q6), m6, atol=1e-7)
    assert torch.allclose(quat_to_matrix3x3(quaternions), expected_matrices, atol=1e-7)
    # matrix3x3 to quat
    assert torch.allclose(matrix3x3_to_quat(m1), q1, atol=1e-7)
    assert torch.allclose(matrix3x3_to_quat(m2), q2, atol=1e-7)
    assert torch.allclose(matrix3x3_to_quat(m3), q3, atol=1e-7)
    assert torch.allclose(matrix3x3_to_quat(m4), q4, atol=1e-7)
    assert torch.allclose(matrix3x3_to_quat(m5), q5AfterMatrix, atol=1e-7)
    assert torch.allclose(matrix3x3_to_quat(m6), q6AfterMatrix, atol=1e-7)
    assert torch.allclose(
        matrix3x3_to_quat(expected_matrices), quaternionsAfterMatrix, atol=1e-7
    )

    # Continuous representation tests
    c1 = torch.tensor([-1, -8.742278e-08, 0, 8.742278e-08, -1, 0])
    c2 = torch.tensor([0.809017, 0, -0.5877853, 0, 1, 0])
    c3 = torch.tensor([1, 0, 0, 0, -1, -8.742278e-08])
    c4 = torch.tensor(
        [-1, -6.181723e-08, -6.181724e-08, 8.742278e-08, -0.7071067, -0.7071068]
    )
    c5 = torch.tensor(
        [5.960464e-08, 8.742278e-08, 0.9999999, 0, -0.9999999, 8.742278e-08]
    )
    c6 = torch.tensor([0.5, -0.8660254, 7.571035e-08, -0.8660254, -0.5, 4.371138e-08])
    expected_c_matrices = torch.stack([c1, c2, c3, c4, c5, c6], 0)
    # continuous representation to quat
    assert torch.allclose(continuous_to_quat(c1), q1, atol=1e-7)
    assert torch.allclose(continuous_to_quat(c2), q2, atol=1e-7)
    assert torch.allclose(continuous_to_quat(c3), q3, atol=1e-7)
    assert torch.allclose(continuous_to_quat(c4), q4, atol=1e-7)
    assert torch.allclose(continuous_to_quat(c5), q5AfterMatrix, atol=1e-7)
    assert torch.allclose(continuous_to_quat(c6), q6AfterMatrix, atol=1e-7)
    assert torch.allclose(
        continuous_to_quat(expected_c_matrices), quaternionsAfterMatrix, atol=1e-7
    )
    # quat to continuous representation
    assert torch.allclose(quat_to_continuous(q1), c1, atol=1e-7)
    assert torch.allclose(quat_to_continuous(q2), c2, atol=1e-7)
    assert torch.allclose(quat_to_continuous(q3), c3, atol=1e-7)
    assert torch.allclose(quat_to_continuous(q4), c4, atol=1e-7)
    assert torch.allclose(quat_to_continuous(q5), c5, atol=1e-7)
    assert torch.allclose(quat_to_continuous(q6), c6, atol=1e-7)
    assert torch.allclose(
        quat_to_continuous(quaternions), expected_c_matrices, atol=1e-7
    )

    # Multiplications tests
    q1q2 = torch.tensor([0.309017, 1.350756e-08, -0.9510565, 4.1572e-08])
    q2q3 = torch.tensor([-0.9510565, 1.350756e-08, 0.309017, 4.1572e-08])
    q3q4 = torch.tensor([-4.038406e-08, -0.9238795, -0.3826835, 1.672763e-08])
    q4q5 = torch.tensor([0.2705981, -0.6532815, -0.2705981, 0.6532815])
    q5q6 = torch.tensor([-0.3535534, -0.6123724, 0.3535534, 0.6123724])
    expected_q_mult = torch.stack([q1q2, q2q3, q3q4, q4q5, q5q6], 0)
    quaternions_mul_1 = torch.stack([q1, q2, q3, q4, q5], 0)
    quaternions_mul_2 = torch.stack([q2, q3, q4, q5, q6], 0)
    # quat to quat
    assert torch.allclose(mul_quat(q1, q2), q1q2, atol=1e-7)
    assert torch.allclose(mul_quat(q2, q3), q2q3, atol=1e-7)
    assert torch.allclose(mul_quat(q3, q4), q3q4, atol=1e-7)
    assert torch.allclose(mul_quat(q4, q5), q4q5, atol=1e-7)
    assert torch.allclose(mul_quat(q5, q6), q5q6, atol=1e-7)
    assert torch.allclose(
        mul_quat(quaternions_mul_1, quaternions_mul_2), expected_q_mult, atol=1e-7
    )
    # rotation matrix to rotation matrix
    m1m2 = torch.tensor(
        [
            -0.809017,
            -7.072651e-08,
            -0.5877853,
            8.742278e-08,
            -1,
            0,
            -0.5877853,
            -5.138582e-08,
            0.809017,
        ]
    )
    m2m3 = torch.tensor(
        [
            0.809017,
            0,
            -0.5877853,
            -5.138582e-08,
            -1,
            -7.072651e-08,
            -0.5877853,
            8.742278e-08,
            -0.809017,
        ]
    )
    m3m4 = torch.tensor(
        [
            -1,
            6.181723e-08,
            6.181725e-08,
            8.742278e-08,
            0.7071066,
            0.7071069,
            0,
            0.7071069,
            -0.7071066,
        ]
    )
    m4m5 = torch.tensor(
        [
            -5.960464e-08,
            -0.7071068,
            0.7071066,
            -8.742277e-08,
            0.7071066,
            0.7071068,
            -0.9999999,
            -1.039641e-07,
            -1.967039e-08,
        ]
    )
    m5m6 = torch.tensor(
        [
            1.055127e-07,
            0.8660253,
            0.4999999,
            -7.907754e-09,
            0.4999999,
            -0.8660254,
            -0.9999999,
            8.742277e-08,
            -5.960465e-08,
        ]
    )
    expected_m_mult = torch.stack([m1m2, m2m3, m3m4, m4m5, m5m6], 0)
    rotation_matrices_mul_1 = torch.stack([m1, m2, m3, m4, m5], 0)
    rotation_matrices_mul_2 = torch.stack([m2, m3, m4, m5, m6], 0)
    # rotation matrix to quat
    assert torch.allclose(torch.abs(mul_rot_mat(m1, m2)), torch.abs(m1m2), atol=1e-6)
    assert torch.allclose(torch.abs(mul_rot_mat(m2, m3)), torch.abs(m2m3), atol=1e-6)
    assert torch.allclose(torch.abs(mul_rot_mat(m3, m4)), torch.abs(m3m4), atol=1e-6)
    assert torch.allclose(torch.abs(mul_rot_mat(m4, m5)), torch.abs(m4m5), atol=1e-6)
    assert torch.allclose(torch.abs(mul_rot_mat(m5, m6)), torch.abs(m5m6), atol=1e-6)
    assert torch.allclose(
        torch.abs(mul_rot_mat(rotation_matrices_mul_1, rotation_matrices_mul_2)),
        torch.abs(expected_m_mult),
        atol=1e-6,
    )


# test_quat_matrix3x3()
