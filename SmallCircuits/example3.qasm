OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
sx q[0];
sx q[1];
rz(pi) q[0];
cx q[0],q[1];
rz(pi) q[0];
