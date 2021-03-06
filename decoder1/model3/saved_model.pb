??#
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68?? 
~
conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_1/kernel
w
!conv_1/kernel/Read/ReadVariableOpReadVariableOpconv_1/kernel*&
_output_shapes
:*
dtype0
n
conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_1/bias
g
conv_1/bias/Read/ReadVariableOpReadVariableOpconv_1/bias*
_output_shapes
:*
dtype0
z
batchnorm_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_1/gamma
s
%batchnorm_1/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_1/gamma*
_output_shapes
:*
dtype0
x
batchnorm_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatchnorm_1/beta
q
$batchnorm_1/beta/Read/ReadVariableOpReadVariableOpbatchnorm_1/beta*
_output_shapes
:*
dtype0
?
batchnorm_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebatchnorm_1/moving_mean

+batchnorm_1/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_1/moving_mean*
_output_shapes
:*
dtype0
?
batchnorm_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatchnorm_1/moving_variance
?
/batchnorm_1/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_1/moving_variance*
_output_shapes
:*
dtype0
~
conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_2/kernel
w
!conv_2/kernel/Read/ReadVariableOpReadVariableOpconv_2/kernel*&
_output_shapes
:*
dtype0
n
conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_2/bias
g
conv_2/bias/Read/ReadVariableOpReadVariableOpconv_2/bias*
_output_shapes
:*
dtype0
z
batchnorm_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_2/gamma
s
%batchnorm_2/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_2/gamma*
_output_shapes
:*
dtype0
x
batchnorm_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatchnorm_2/beta
q
$batchnorm_2/beta/Read/ReadVariableOpReadVariableOpbatchnorm_2/beta*
_output_shapes
:*
dtype0
?
batchnorm_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebatchnorm_2/moving_mean

+batchnorm_2/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_2/moving_mean*
_output_shapes
:*
dtype0
?
batchnorm_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatchnorm_2/moving_variance
?
/batchnorm_2/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_2/moving_variance*
_output_shapes
:*
dtype0
~
conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_3/kernel
w
!conv_3/kernel/Read/ReadVariableOpReadVariableOpconv_3/kernel*&
_output_shapes
:*
dtype0
n
conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_3/bias
g
conv_3/bias/Read/ReadVariableOpReadVariableOpconv_3/bias*
_output_shapes
:*
dtype0
z
batchnorm_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_3/gamma
s
%batchnorm_3/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_3/gamma*
_output_shapes
:*
dtype0
x
batchnorm_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatchnorm_3/beta
q
$batchnorm_3/beta/Read/ReadVariableOpReadVariableOpbatchnorm_3/beta*
_output_shapes
:*
dtype0
?
batchnorm_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebatchnorm_3/moving_mean

+batchnorm_3/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_3/moving_mean*
_output_shapes
:*
dtype0
?
batchnorm_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatchnorm_3/moving_variance
?
/batchnorm_3/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_3/moving_variance*
_output_shapes
:*
dtype0
?
conv_4a/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_4a/kernel
y
"conv_4a/kernel/Read/ReadVariableOpReadVariableOpconv_4a/kernel*&
_output_shapes
:*
dtype0
p
conv_4a/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_4a/bias
i
 conv_4a/bias/Read/ReadVariableOpReadVariableOpconv_4a/bias*
_output_shapes
:*
dtype0
|
batchnorm_4a/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namebatchnorm_4a/gamma
u
&batchnorm_4a/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_4a/gamma*
_output_shapes
:*
dtype0
z
batchnorm_4a/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_4a/beta
s
%batchnorm_4a/beta/Read/ReadVariableOpReadVariableOpbatchnorm_4a/beta*
_output_shapes
:*
dtype0
?
batchnorm_4a/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatchnorm_4a/moving_mean
?
,batchnorm_4a/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_4a/moving_mean*
_output_shapes
:*
dtype0
?
batchnorm_4a/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatchnorm_4a/moving_variance
?
0batchnorm_4a/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_4a/moving_variance*
_output_shapes
:*
dtype0
?
conv_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv_transpose_1/kernel
?
+conv_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv_transpose_1/kernel*&
_output_shapes
:*
dtype0
?
conv_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv_transpose_1/bias
{
)conv_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv_transpose_1/bias*
_output_shapes
:*
dtype0
z
batchnorm_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_4/gamma
s
%batchnorm_4/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_4/gamma*
_output_shapes
:*
dtype0
x
batchnorm_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatchnorm_4/beta
q
$batchnorm_4/beta/Read/ReadVariableOpReadVariableOpbatchnorm_4/beta*
_output_shapes
:*
dtype0
?
batchnorm_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebatchnorm_4/moving_mean

+batchnorm_4/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_4/moving_mean*
_output_shapes
:*
dtype0
?
batchnorm_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatchnorm_4/moving_variance
?
/batchnorm_4/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_4/moving_variance*
_output_shapes
:*
dtype0
?
conv_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv_transpose_2/kernel
?
+conv_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv_transpose_2/kernel*&
_output_shapes
:*
dtype0
?
conv_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv_transpose_2/bias
{
)conv_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv_transpose_2/bias*
_output_shapes
:*
dtype0
z
batchnorm_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_5/gamma
s
%batchnorm_5/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_5/gamma*
_output_shapes
:*
dtype0
x
batchnorm_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatchnorm_5/beta
q
$batchnorm_5/beta/Read/ReadVariableOpReadVariableOpbatchnorm_5/beta*
_output_shapes
:*
dtype0
?
batchnorm_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebatchnorm_5/moving_mean

+batchnorm_5/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_5/moving_mean*
_output_shapes
:*
dtype0
?
batchnorm_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatchnorm_5/moving_variance
?
/batchnorm_5/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_5/moving_variance*
_output_shapes
:*
dtype0
?
conv_transpose_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv_transpose_30/kernel
?
,conv_transpose_30/kernel/Read/ReadVariableOpReadVariableOpconv_transpose_30/kernel*&
_output_shapes
:*
dtype0
?
conv_transpose_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameconv_transpose_30/bias
}
*conv_transpose_30/bias/Read/ReadVariableOpReadVariableOpconv_transpose_30/bias*
_output_shapes
:*
dtype0
|
batchnorm_31/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namebatchnorm_31/gamma
u
&batchnorm_31/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_31/gamma*
_output_shapes
:*
dtype0
z
batchnorm_31/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_31/beta
s
%batchnorm_31/beta/Read/ReadVariableOpReadVariableOpbatchnorm_31/beta*
_output_shapes
:*
dtype0
?
batchnorm_31/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatchnorm_31/moving_mean
?
,batchnorm_31/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_31/moving_mean*
_output_shapes
:*
dtype0
?
batchnorm_31/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatchnorm_31/moving_variance
?
0batchnorm_31/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_31/moving_variance*
_output_shapes
:*
dtype0
?
conv_transpose_a/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv_transpose_a/kernel
?
+conv_transpose_a/kernel/Read/ReadVariableOpReadVariableOpconv_transpose_a/kernel*&
_output_shapes
:*
dtype0
?
conv_transpose_a/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv_transpose_a/bias
{
)conv_transpose_a/bias/Read/ReadVariableOpReadVariableOpconv_transpose_a/bias*
_output_shapes
:*
dtype0
z
batchnorm_b/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_b/gamma
s
%batchnorm_b/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_b/gamma*
_output_shapes
:*
dtype0
x
batchnorm_b/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatchnorm_b/beta
q
$batchnorm_b/beta/Read/ReadVariableOpReadVariableOpbatchnorm_b/beta*
_output_shapes
:*
dtype0
?
batchnorm_b/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebatchnorm_b/moving_mean

+batchnorm_b/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_b/moving_mean*
_output_shapes
:*
dtype0
?
batchnorm_b/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatchnorm_b/moving_variance
?
/batchnorm_b/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_b/moving_variance*
_output_shapes
:*
dtype0
?
conv_transpose_4a/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv_transpose_4a/kernel
?
,conv_transpose_4a/kernel/Read/ReadVariableOpReadVariableOpconv_transpose_4a/kernel*&
_output_shapes
:*
dtype0
?
conv_transpose_4a/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameconv_transpose_4a/bias
}
*conv_transpose_4a/bias/Read/ReadVariableOpReadVariableOpconv_transpose_4a/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv_1/kernel/m
{
#conv_1/kernel/m/Read/ReadVariableOpReadVariableOpconv_1/kernel/m*&
_output_shapes
:*
dtype0
r
conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_1/bias/m
k
!conv_1/bias/m/Read/ReadVariableOpReadVariableOpconv_1/bias/m*
_output_shapes
:*
dtype0
~
batchnorm_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namebatchnorm_1/gamma/m
w
'batchnorm_1/gamma/m/Read/ReadVariableOpReadVariableOpbatchnorm_1/gamma/m*
_output_shapes
:*
dtype0
|
batchnorm_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namebatchnorm_1/beta/m
u
&batchnorm_1/beta/m/Read/ReadVariableOpReadVariableOpbatchnorm_1/beta/m*
_output_shapes
:*
dtype0
?
conv_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv_2/kernel/m
{
#conv_2/kernel/m/Read/ReadVariableOpReadVariableOpconv_2/kernel/m*&
_output_shapes
:*
dtype0
r
conv_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_2/bias/m
k
!conv_2/bias/m/Read/ReadVariableOpReadVariableOpconv_2/bias/m*
_output_shapes
:*
dtype0
~
batchnorm_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namebatchnorm_2/gamma/m
w
'batchnorm_2/gamma/m/Read/ReadVariableOpReadVariableOpbatchnorm_2/gamma/m*
_output_shapes
:*
dtype0
|
batchnorm_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namebatchnorm_2/beta/m
u
&batchnorm_2/beta/m/Read/ReadVariableOpReadVariableOpbatchnorm_2/beta/m*
_output_shapes
:*
dtype0
?
conv_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv_3/kernel/m
{
#conv_3/kernel/m/Read/ReadVariableOpReadVariableOpconv_3/kernel/m*&
_output_shapes
:*
dtype0
r
conv_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_3/bias/m
k
!conv_3/bias/m/Read/ReadVariableOpReadVariableOpconv_3/bias/m*
_output_shapes
:*
dtype0
~
batchnorm_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namebatchnorm_3/gamma/m
w
'batchnorm_3/gamma/m/Read/ReadVariableOpReadVariableOpbatchnorm_3/gamma/m*
_output_shapes
:*
dtype0
|
batchnorm_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namebatchnorm_3/beta/m
u
&batchnorm_3/beta/m/Read/ReadVariableOpReadVariableOpbatchnorm_3/beta/m*
_output_shapes
:*
dtype0
?
conv_4a/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv_4a/kernel/m
}
$conv_4a/kernel/m/Read/ReadVariableOpReadVariableOpconv_4a/kernel/m*&
_output_shapes
:*
dtype0
t
conv_4a/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_4a/bias/m
m
"conv_4a/bias/m/Read/ReadVariableOpReadVariableOpconv_4a/bias/m*
_output_shapes
:*
dtype0
?
batchnorm_4a/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namebatchnorm_4a/gamma/m
y
(batchnorm_4a/gamma/m/Read/ReadVariableOpReadVariableOpbatchnorm_4a/gamma/m*
_output_shapes
:*
dtype0
~
batchnorm_4a/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namebatchnorm_4a/beta/m
w
'batchnorm_4a/beta/m/Read/ReadVariableOpReadVariableOpbatchnorm_4a/beta/m*
_output_shapes
:*
dtype0
?
conv_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv_transpose_1/kernel/m
?
-conv_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOpconv_transpose_1/kernel/m*&
_output_shapes
:*
dtype0
?
conv_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv_transpose_1/bias/m

+conv_transpose_1/bias/m/Read/ReadVariableOpReadVariableOpconv_transpose_1/bias/m*
_output_shapes
:*
dtype0
~
batchnorm_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namebatchnorm_4/gamma/m
w
'batchnorm_4/gamma/m/Read/ReadVariableOpReadVariableOpbatchnorm_4/gamma/m*
_output_shapes
:*
dtype0
|
batchnorm_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namebatchnorm_4/beta/m
u
&batchnorm_4/beta/m/Read/ReadVariableOpReadVariableOpbatchnorm_4/beta/m*
_output_shapes
:*
dtype0
?
conv_transpose_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv_transpose_2/kernel/m
?
-conv_transpose_2/kernel/m/Read/ReadVariableOpReadVariableOpconv_transpose_2/kernel/m*&
_output_shapes
:*
dtype0
?
conv_transpose_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv_transpose_2/bias/m

+conv_transpose_2/bias/m/Read/ReadVariableOpReadVariableOpconv_transpose_2/bias/m*
_output_shapes
:*
dtype0
~
batchnorm_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namebatchnorm_5/gamma/m
w
'batchnorm_5/gamma/m/Read/ReadVariableOpReadVariableOpbatchnorm_5/gamma/m*
_output_shapes
:*
dtype0
|
batchnorm_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namebatchnorm_5/beta/m
u
&batchnorm_5/beta/m/Read/ReadVariableOpReadVariableOpbatchnorm_5/beta/m*
_output_shapes
:*
dtype0
?
conv_transpose_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv_transpose_30/kernel/m
?
.conv_transpose_30/kernel/m/Read/ReadVariableOpReadVariableOpconv_transpose_30/kernel/m*&
_output_shapes
:*
dtype0
?
conv_transpose_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv_transpose_30/bias/m
?
,conv_transpose_30/bias/m/Read/ReadVariableOpReadVariableOpconv_transpose_30/bias/m*
_output_shapes
:*
dtype0
?
batchnorm_31/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namebatchnorm_31/gamma/m
y
(batchnorm_31/gamma/m/Read/ReadVariableOpReadVariableOpbatchnorm_31/gamma/m*
_output_shapes
:*
dtype0
~
batchnorm_31/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namebatchnorm_31/beta/m
w
'batchnorm_31/beta/m/Read/ReadVariableOpReadVariableOpbatchnorm_31/beta/m*
_output_shapes
:*
dtype0
?
conv_transpose_a/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv_transpose_a/kernel/m
?
-conv_transpose_a/kernel/m/Read/ReadVariableOpReadVariableOpconv_transpose_a/kernel/m*&
_output_shapes
:*
dtype0
?
conv_transpose_a/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv_transpose_a/bias/m

+conv_transpose_a/bias/m/Read/ReadVariableOpReadVariableOpconv_transpose_a/bias/m*
_output_shapes
:*
dtype0
~
batchnorm_b/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namebatchnorm_b/gamma/m
w
'batchnorm_b/gamma/m/Read/ReadVariableOpReadVariableOpbatchnorm_b/gamma/m*
_output_shapes
:*
dtype0
|
batchnorm_b/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namebatchnorm_b/beta/m
u
&batchnorm_b/beta/m/Read/ReadVariableOpReadVariableOpbatchnorm_b/beta/m*
_output_shapes
:*
dtype0
?
conv_transpose_4a/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv_transpose_4a/kernel/m
?
.conv_transpose_4a/kernel/m/Read/ReadVariableOpReadVariableOpconv_transpose_4a/kernel/m*&
_output_shapes
:*
dtype0
?
conv_transpose_4a/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv_transpose_4a/bias/m
?
,conv_transpose_4a/bias/m/Read/ReadVariableOpReadVariableOpconv_transpose_4a/bias/m*
_output_shapes
:*
dtype0
?
conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv_1/kernel/v
{
#conv_1/kernel/v/Read/ReadVariableOpReadVariableOpconv_1/kernel/v*&
_output_shapes
:*
dtype0
r
conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_1/bias/v
k
!conv_1/bias/v/Read/ReadVariableOpReadVariableOpconv_1/bias/v*
_output_shapes
:*
dtype0
~
batchnorm_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namebatchnorm_1/gamma/v
w
'batchnorm_1/gamma/v/Read/ReadVariableOpReadVariableOpbatchnorm_1/gamma/v*
_output_shapes
:*
dtype0
|
batchnorm_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namebatchnorm_1/beta/v
u
&batchnorm_1/beta/v/Read/ReadVariableOpReadVariableOpbatchnorm_1/beta/v*
_output_shapes
:*
dtype0
?
conv_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv_2/kernel/v
{
#conv_2/kernel/v/Read/ReadVariableOpReadVariableOpconv_2/kernel/v*&
_output_shapes
:*
dtype0
r
conv_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_2/bias/v
k
!conv_2/bias/v/Read/ReadVariableOpReadVariableOpconv_2/bias/v*
_output_shapes
:*
dtype0
~
batchnorm_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namebatchnorm_2/gamma/v
w
'batchnorm_2/gamma/v/Read/ReadVariableOpReadVariableOpbatchnorm_2/gamma/v*
_output_shapes
:*
dtype0
|
batchnorm_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namebatchnorm_2/beta/v
u
&batchnorm_2/beta/v/Read/ReadVariableOpReadVariableOpbatchnorm_2/beta/v*
_output_shapes
:*
dtype0
?
conv_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv_3/kernel/v
{
#conv_3/kernel/v/Read/ReadVariableOpReadVariableOpconv_3/kernel/v*&
_output_shapes
:*
dtype0
r
conv_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_3/bias/v
k
!conv_3/bias/v/Read/ReadVariableOpReadVariableOpconv_3/bias/v*
_output_shapes
:*
dtype0
~
batchnorm_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namebatchnorm_3/gamma/v
w
'batchnorm_3/gamma/v/Read/ReadVariableOpReadVariableOpbatchnorm_3/gamma/v*
_output_shapes
:*
dtype0
|
batchnorm_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namebatchnorm_3/beta/v
u
&batchnorm_3/beta/v/Read/ReadVariableOpReadVariableOpbatchnorm_3/beta/v*
_output_shapes
:*
dtype0
?
conv_4a/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv_4a/kernel/v
}
$conv_4a/kernel/v/Read/ReadVariableOpReadVariableOpconv_4a/kernel/v*&
_output_shapes
:*
dtype0
t
conv_4a/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_4a/bias/v
m
"conv_4a/bias/v/Read/ReadVariableOpReadVariableOpconv_4a/bias/v*
_output_shapes
:*
dtype0
?
batchnorm_4a/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namebatchnorm_4a/gamma/v
y
(batchnorm_4a/gamma/v/Read/ReadVariableOpReadVariableOpbatchnorm_4a/gamma/v*
_output_shapes
:*
dtype0
~
batchnorm_4a/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namebatchnorm_4a/beta/v
w
'batchnorm_4a/beta/v/Read/ReadVariableOpReadVariableOpbatchnorm_4a/beta/v*
_output_shapes
:*
dtype0
?
conv_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv_transpose_1/kernel/v
?
-conv_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOpconv_transpose_1/kernel/v*&
_output_shapes
:*
dtype0
?
conv_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv_transpose_1/bias/v

+conv_transpose_1/bias/v/Read/ReadVariableOpReadVariableOpconv_transpose_1/bias/v*
_output_shapes
:*
dtype0
~
batchnorm_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namebatchnorm_4/gamma/v
w
'batchnorm_4/gamma/v/Read/ReadVariableOpReadVariableOpbatchnorm_4/gamma/v*
_output_shapes
:*
dtype0
|
batchnorm_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namebatchnorm_4/beta/v
u
&batchnorm_4/beta/v/Read/ReadVariableOpReadVariableOpbatchnorm_4/beta/v*
_output_shapes
:*
dtype0
?
conv_transpose_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv_transpose_2/kernel/v
?
-conv_transpose_2/kernel/v/Read/ReadVariableOpReadVariableOpconv_transpose_2/kernel/v*&
_output_shapes
:*
dtype0
?
conv_transpose_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv_transpose_2/bias/v

+conv_transpose_2/bias/v/Read/ReadVariableOpReadVariableOpconv_transpose_2/bias/v*
_output_shapes
:*
dtype0
~
batchnorm_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namebatchnorm_5/gamma/v
w
'batchnorm_5/gamma/v/Read/ReadVariableOpReadVariableOpbatchnorm_5/gamma/v*
_output_shapes
:*
dtype0
|
batchnorm_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namebatchnorm_5/beta/v
u
&batchnorm_5/beta/v/Read/ReadVariableOpReadVariableOpbatchnorm_5/beta/v*
_output_shapes
:*
dtype0
?
conv_transpose_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv_transpose_30/kernel/v
?
.conv_transpose_30/kernel/v/Read/ReadVariableOpReadVariableOpconv_transpose_30/kernel/v*&
_output_shapes
:*
dtype0
?
conv_transpose_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv_transpose_30/bias/v
?
,conv_transpose_30/bias/v/Read/ReadVariableOpReadVariableOpconv_transpose_30/bias/v*
_output_shapes
:*
dtype0
?
batchnorm_31/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namebatchnorm_31/gamma/v
y
(batchnorm_31/gamma/v/Read/ReadVariableOpReadVariableOpbatchnorm_31/gamma/v*
_output_shapes
:*
dtype0
~
batchnorm_31/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namebatchnorm_31/beta/v
w
'batchnorm_31/beta/v/Read/ReadVariableOpReadVariableOpbatchnorm_31/beta/v*
_output_shapes
:*
dtype0
?
conv_transpose_a/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv_transpose_a/kernel/v
?
-conv_transpose_a/kernel/v/Read/ReadVariableOpReadVariableOpconv_transpose_a/kernel/v*&
_output_shapes
:*
dtype0
?
conv_transpose_a/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv_transpose_a/bias/v

+conv_transpose_a/bias/v/Read/ReadVariableOpReadVariableOpconv_transpose_a/bias/v*
_output_shapes
:*
dtype0
~
batchnorm_b/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namebatchnorm_b/gamma/v
w
'batchnorm_b/gamma/v/Read/ReadVariableOpReadVariableOpbatchnorm_b/gamma/v*
_output_shapes
:*
dtype0
|
batchnorm_b/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namebatchnorm_b/beta/v
u
&batchnorm_b/beta/v/Read/ReadVariableOpReadVariableOpbatchnorm_b/beta/v*
_output_shapes
:*
dtype0
?
conv_transpose_4a/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv_transpose_4a/kernel/v
?
.conv_transpose_4a/kernel/v/Read/ReadVariableOpReadVariableOpconv_transpose_4a/kernel/v*&
_output_shapes
:*
dtype0
?
conv_transpose_4a/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv_transpose_4a/bias/v
?
,conv_transpose_4a/bias/v/Read/ReadVariableOpReadVariableOpconv_transpose_4a/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer_with_weights-12
layer-19
layer_with_weights-13
layer-20
layer-21
layer_with_weights-14
layer-22
layer_with_weights-15
layer-23
layer-24
layer_with_weights-16
layer-25
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_default_save_signature
#
signatures*
* 
?

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses*
?
,axis
	-gamma
.beta
/moving_mean
0moving_variance
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses*
?
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses* 
?

=kernel
>bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses*
?
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses*
?
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses* 
?

Vkernel
Wbias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses*
?
^axis
	_gamma
`beta
amoving_mean
bmoving_variance
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses*
?
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses* 
?

okernel
pbias
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses*
?
waxis
	xgamma
ybeta
zmoving_mean
{moving_variance
|	variables
}trainable_variables
~regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?$m?%m?-m?.m?=m?>m?Fm?Gm?Vm?Wm?_m?`m?om?pm?xm?ym?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?$v?%v?-v?.v?=v?>v?Fv?Gv?Vv?Wv?_v?`v?ov?pv?xv?yv?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?*
?
$0
%1
-2
.3
/4
05
=6
>7
F8
G9
H10
I11
V12
W13
_14
`15
a16
b17
o18
p19
x20
y21
z22
{23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49*
?
$0
%1
-2
.3
=4
>5
F6
G7
V8
W9
_10
`11
o12
p13
x14
y15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
"_default_save_signature
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
* 

?serving_default* 
]W
VARIABLE_VALUEconv_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
* 
* 
* 
`Z
VARIABLE_VALUEbatchnorm_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbatchnorm_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbatchnorm_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatchnorm_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
-0
.1
/2
03*

-0
.1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses* 
* 
* 
]W
VARIABLE_VALUEconv_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

=0
>1*

=0
>1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*
* 
* 
* 
`Z
VARIABLE_VALUEbatchnorm_2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbatchnorm_2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbatchnorm_2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatchnorm_2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
F0
G1
H2
I3*

F0
G1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses* 
* 
* 
]W
VARIABLE_VALUEconv_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

V0
W1*

V0
W1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*
* 
* 
* 
`Z
VARIABLE_VALUEbatchnorm_3/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbatchnorm_3/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbatchnorm_3/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatchnorm_3/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
_0
`1
a2
b3*

_0
`1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses* 
* 
* 
^X
VARIABLE_VALUEconv_4a/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv_4a/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

o0
p1*

o0
p1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*
* 
* 
* 
a[
VARIABLE_VALUEbatchnorm_4a/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEbatchnorm_4a/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEbatchnorm_4a/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEbatchnorm_4a/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
x0
y1
z2
{3*

x0
y1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
|	variables
}trainable_variables
~regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
ga
VARIABLE_VALUEconv_transpose_1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEconv_transpose_1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
`Z
VARIABLE_VALUEbatchnorm_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbatchnorm_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEbatchnorm_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatchnorm_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
?0
?1
?2
?3*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
hb
VARIABLE_VALUEconv_transpose_2/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEconv_transpose_2/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
a[
VARIABLE_VALUEbatchnorm_5/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEbatchnorm_5/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEbatchnorm_5/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEbatchnorm_5/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
?0
?1
?2
?3*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
ic
VARIABLE_VALUEconv_transpose_30/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEconv_transpose_30/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
b\
VARIABLE_VALUEbatchnorm_31/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEbatchnorm_31/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEbatchnorm_31/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEbatchnorm_31/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
?0
?1
?2
?3*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
hb
VARIABLE_VALUEconv_transpose_a/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEconv_transpose_a/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
a[
VARIABLE_VALUEbatchnorm_b/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEbatchnorm_b/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEbatchnorm_b/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEbatchnorm_b/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
?0
?1
?2
?3*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
ic
VARIABLE_VALUEconv_transpose_4a/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEconv_transpose_4a/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
?
/0
01
H2
I3
a4
b5
z6
{7
?8
?9
?10
?11
?12
?13
?14
?15*
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25*

?0*
* 
* 
* 
* 
* 
* 
* 
* 

/0
01*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

H0
I1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

a0
b1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

z0
{1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

?total

?count
?	variables
?	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
{u
VARIABLE_VALUEconv_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEconv_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEbatchnorm_1/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEbatchnorm_1/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEconv_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEconv_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEbatchnorm_2/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEbatchnorm_2/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEconv_3/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEconv_3/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEbatchnorm_3/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEbatchnorm_3/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEconv_4a/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEconv_4a/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEbatchnorm_4a/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEbatchnorm_4a/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEconv_transpose_1/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEconv_transpose_1/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEbatchnorm_4/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEbatchnorm_4/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEconv_transpose_2/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEconv_transpose_2/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEbatchnorm_5/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEbatchnorm_5/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEconv_transpose_30/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEconv_transpose_30/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEbatchnorm_31/gamma/mRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEbatchnorm_31/beta/mQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEconv_transpose_a/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEconv_transpose_a/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEbatchnorm_b/gamma/mRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEbatchnorm_b/beta/mQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEconv_transpose_4a/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEconv_transpose_4a/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEconv_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEconv_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEbatchnorm_1/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEbatchnorm_1/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEconv_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEconv_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEbatchnorm_2/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEbatchnorm_2/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEconv_3/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEconv_3/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEbatchnorm_3/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEbatchnorm_3/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEconv_4a/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEconv_4a/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEbatchnorm_4a/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEbatchnorm_4a/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEconv_transpose_1/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEconv_transpose_1/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEbatchnorm_4/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEbatchnorm_4/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEconv_transpose_2/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEconv_transpose_2/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEbatchnorm_5/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEbatchnorm_5/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEconv_transpose_30/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEconv_transpose_30/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEbatchnorm_31/gamma/vRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEbatchnorm_31/beta/vQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEconv_transpose_a/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEconv_transpose_a/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEbatchnorm_b/gamma/vRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEbatchnorm_b/beta/vQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEconv_transpose_4a/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEconv_transpose_4a/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_input_layerPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerconv_1/kernelconv_1/biasbatchnorm_1/gammabatchnorm_1/betabatchnorm_1/moving_meanbatchnorm_1/moving_varianceconv_2/kernelconv_2/biasbatchnorm_2/gammabatchnorm_2/betabatchnorm_2/moving_meanbatchnorm_2/moving_varianceconv_3/kernelconv_3/biasbatchnorm_3/gammabatchnorm_3/betabatchnorm_3/moving_meanbatchnorm_3/moving_varianceconv_4a/kernelconv_4a/biasbatchnorm_4a/gammabatchnorm_4a/betabatchnorm_4a/moving_meanbatchnorm_4a/moving_varianceconv_transpose_1/kernelconv_transpose_1/biasbatchnorm_4/gammabatchnorm_4/betabatchnorm_4/moving_meanbatchnorm_4/moving_varianceconv_transpose_2/kernelconv_transpose_2/biasbatchnorm_5/gammabatchnorm_5/betabatchnorm_5/moving_meanbatchnorm_5/moving_varianceconv_transpose_30/kernelconv_transpose_30/biasbatchnorm_31/gammabatchnorm_31/betabatchnorm_31/moving_meanbatchnorm_31/moving_varianceconv_transpose_a/kernelconv_transpose_a/biasbatchnorm_b/gammabatchnorm_b/betabatchnorm_b/moving_meanbatchnorm_b/moving_varianceconv_transpose_4a/kernelconv_transpose_4a/bias*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_4148
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?)
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv_1/kernel/Read/ReadVariableOpconv_1/bias/Read/ReadVariableOp%batchnorm_1/gamma/Read/ReadVariableOp$batchnorm_1/beta/Read/ReadVariableOp+batchnorm_1/moving_mean/Read/ReadVariableOp/batchnorm_1/moving_variance/Read/ReadVariableOp!conv_2/kernel/Read/ReadVariableOpconv_2/bias/Read/ReadVariableOp%batchnorm_2/gamma/Read/ReadVariableOp$batchnorm_2/beta/Read/ReadVariableOp+batchnorm_2/moving_mean/Read/ReadVariableOp/batchnorm_2/moving_variance/Read/ReadVariableOp!conv_3/kernel/Read/ReadVariableOpconv_3/bias/Read/ReadVariableOp%batchnorm_3/gamma/Read/ReadVariableOp$batchnorm_3/beta/Read/ReadVariableOp+batchnorm_3/moving_mean/Read/ReadVariableOp/batchnorm_3/moving_variance/Read/ReadVariableOp"conv_4a/kernel/Read/ReadVariableOp conv_4a/bias/Read/ReadVariableOp&batchnorm_4a/gamma/Read/ReadVariableOp%batchnorm_4a/beta/Read/ReadVariableOp,batchnorm_4a/moving_mean/Read/ReadVariableOp0batchnorm_4a/moving_variance/Read/ReadVariableOp+conv_transpose_1/kernel/Read/ReadVariableOp)conv_transpose_1/bias/Read/ReadVariableOp%batchnorm_4/gamma/Read/ReadVariableOp$batchnorm_4/beta/Read/ReadVariableOp+batchnorm_4/moving_mean/Read/ReadVariableOp/batchnorm_4/moving_variance/Read/ReadVariableOp+conv_transpose_2/kernel/Read/ReadVariableOp)conv_transpose_2/bias/Read/ReadVariableOp%batchnorm_5/gamma/Read/ReadVariableOp$batchnorm_5/beta/Read/ReadVariableOp+batchnorm_5/moving_mean/Read/ReadVariableOp/batchnorm_5/moving_variance/Read/ReadVariableOp,conv_transpose_30/kernel/Read/ReadVariableOp*conv_transpose_30/bias/Read/ReadVariableOp&batchnorm_31/gamma/Read/ReadVariableOp%batchnorm_31/beta/Read/ReadVariableOp,batchnorm_31/moving_mean/Read/ReadVariableOp0batchnorm_31/moving_variance/Read/ReadVariableOp+conv_transpose_a/kernel/Read/ReadVariableOp)conv_transpose_a/bias/Read/ReadVariableOp%batchnorm_b/gamma/Read/ReadVariableOp$batchnorm_b/beta/Read/ReadVariableOp+batchnorm_b/moving_mean/Read/ReadVariableOp/batchnorm_b/moving_variance/Read/ReadVariableOp,conv_transpose_4a/kernel/Read/ReadVariableOp*conv_transpose_4a/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp#conv_1/kernel/m/Read/ReadVariableOp!conv_1/bias/m/Read/ReadVariableOp'batchnorm_1/gamma/m/Read/ReadVariableOp&batchnorm_1/beta/m/Read/ReadVariableOp#conv_2/kernel/m/Read/ReadVariableOp!conv_2/bias/m/Read/ReadVariableOp'batchnorm_2/gamma/m/Read/ReadVariableOp&batchnorm_2/beta/m/Read/ReadVariableOp#conv_3/kernel/m/Read/ReadVariableOp!conv_3/bias/m/Read/ReadVariableOp'batchnorm_3/gamma/m/Read/ReadVariableOp&batchnorm_3/beta/m/Read/ReadVariableOp$conv_4a/kernel/m/Read/ReadVariableOp"conv_4a/bias/m/Read/ReadVariableOp(batchnorm_4a/gamma/m/Read/ReadVariableOp'batchnorm_4a/beta/m/Read/ReadVariableOp-conv_transpose_1/kernel/m/Read/ReadVariableOp+conv_transpose_1/bias/m/Read/ReadVariableOp'batchnorm_4/gamma/m/Read/ReadVariableOp&batchnorm_4/beta/m/Read/ReadVariableOp-conv_transpose_2/kernel/m/Read/ReadVariableOp+conv_transpose_2/bias/m/Read/ReadVariableOp'batchnorm_5/gamma/m/Read/ReadVariableOp&batchnorm_5/beta/m/Read/ReadVariableOp.conv_transpose_30/kernel/m/Read/ReadVariableOp,conv_transpose_30/bias/m/Read/ReadVariableOp(batchnorm_31/gamma/m/Read/ReadVariableOp'batchnorm_31/beta/m/Read/ReadVariableOp-conv_transpose_a/kernel/m/Read/ReadVariableOp+conv_transpose_a/bias/m/Read/ReadVariableOp'batchnorm_b/gamma/m/Read/ReadVariableOp&batchnorm_b/beta/m/Read/ReadVariableOp.conv_transpose_4a/kernel/m/Read/ReadVariableOp,conv_transpose_4a/bias/m/Read/ReadVariableOp#conv_1/kernel/v/Read/ReadVariableOp!conv_1/bias/v/Read/ReadVariableOp'batchnorm_1/gamma/v/Read/ReadVariableOp&batchnorm_1/beta/v/Read/ReadVariableOp#conv_2/kernel/v/Read/ReadVariableOp!conv_2/bias/v/Read/ReadVariableOp'batchnorm_2/gamma/v/Read/ReadVariableOp&batchnorm_2/beta/v/Read/ReadVariableOp#conv_3/kernel/v/Read/ReadVariableOp!conv_3/bias/v/Read/ReadVariableOp'batchnorm_3/gamma/v/Read/ReadVariableOp&batchnorm_3/beta/v/Read/ReadVariableOp$conv_4a/kernel/v/Read/ReadVariableOp"conv_4a/bias/v/Read/ReadVariableOp(batchnorm_4a/gamma/v/Read/ReadVariableOp'batchnorm_4a/beta/v/Read/ReadVariableOp-conv_transpose_1/kernel/v/Read/ReadVariableOp+conv_transpose_1/bias/v/Read/ReadVariableOp'batchnorm_4/gamma/v/Read/ReadVariableOp&batchnorm_4/beta/v/Read/ReadVariableOp-conv_transpose_2/kernel/v/Read/ReadVariableOp+conv_transpose_2/bias/v/Read/ReadVariableOp'batchnorm_5/gamma/v/Read/ReadVariableOp&batchnorm_5/beta/v/Read/ReadVariableOp.conv_transpose_30/kernel/v/Read/ReadVariableOp,conv_transpose_30/bias/v/Read/ReadVariableOp(batchnorm_31/gamma/v/Read/ReadVariableOp'batchnorm_31/beta/v/Read/ReadVariableOp-conv_transpose_a/kernel/v/Read/ReadVariableOp+conv_transpose_a/bias/v/Read/ReadVariableOp'batchnorm_b/gamma/v/Read/ReadVariableOp&batchnorm_b/beta/v/Read/ReadVariableOp.conv_transpose_4a/kernel/v/Read/ReadVariableOp,conv_transpose_4a/bias/v/Read/ReadVariableOpConst*?
Tin~
|2z*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference__traced_save_5394
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_1/kernelconv_1/biasbatchnorm_1/gammabatchnorm_1/betabatchnorm_1/moving_meanbatchnorm_1/moving_varianceconv_2/kernelconv_2/biasbatchnorm_2/gammabatchnorm_2/betabatchnorm_2/moving_meanbatchnorm_2/moving_varianceconv_3/kernelconv_3/biasbatchnorm_3/gammabatchnorm_3/betabatchnorm_3/moving_meanbatchnorm_3/moving_varianceconv_4a/kernelconv_4a/biasbatchnorm_4a/gammabatchnorm_4a/betabatchnorm_4a/moving_meanbatchnorm_4a/moving_varianceconv_transpose_1/kernelconv_transpose_1/biasbatchnorm_4/gammabatchnorm_4/betabatchnorm_4/moving_meanbatchnorm_4/moving_varianceconv_transpose_2/kernelconv_transpose_2/biasbatchnorm_5/gammabatchnorm_5/betabatchnorm_5/moving_meanbatchnorm_5/moving_varianceconv_transpose_30/kernelconv_transpose_30/biasbatchnorm_31/gammabatchnorm_31/betabatchnorm_31/moving_meanbatchnorm_31/moving_varianceconv_transpose_a/kernelconv_transpose_a/biasbatchnorm_b/gammabatchnorm_b/betabatchnorm_b/moving_meanbatchnorm_b/moving_varianceconv_transpose_4a/kernelconv_transpose_4a/biastotalcountconv_1/kernel/mconv_1/bias/mbatchnorm_1/gamma/mbatchnorm_1/beta/mconv_2/kernel/mconv_2/bias/mbatchnorm_2/gamma/mbatchnorm_2/beta/mconv_3/kernel/mconv_3/bias/mbatchnorm_3/gamma/mbatchnorm_3/beta/mconv_4a/kernel/mconv_4a/bias/mbatchnorm_4a/gamma/mbatchnorm_4a/beta/mconv_transpose_1/kernel/mconv_transpose_1/bias/mbatchnorm_4/gamma/mbatchnorm_4/beta/mconv_transpose_2/kernel/mconv_transpose_2/bias/mbatchnorm_5/gamma/mbatchnorm_5/beta/mconv_transpose_30/kernel/mconv_transpose_30/bias/mbatchnorm_31/gamma/mbatchnorm_31/beta/mconv_transpose_a/kernel/mconv_transpose_a/bias/mbatchnorm_b/gamma/mbatchnorm_b/beta/mconv_transpose_4a/kernel/mconv_transpose_4a/bias/mconv_1/kernel/vconv_1/bias/vbatchnorm_1/gamma/vbatchnorm_1/beta/vconv_2/kernel/vconv_2/bias/vbatchnorm_2/gamma/vbatchnorm_2/beta/vconv_3/kernel/vconv_3/bias/vbatchnorm_3/gamma/vbatchnorm_3/beta/vconv_4a/kernel/vconv_4a/bias/vbatchnorm_4a/gamma/vbatchnorm_4a/beta/vconv_transpose_1/kernel/vconv_transpose_1/bias/vbatchnorm_4/gamma/vbatchnorm_4/beta/vconv_transpose_2/kernel/vconv_transpose_2/bias/vbatchnorm_5/gamma/vbatchnorm_5/beta/vconv_transpose_30/kernel/vconv_transpose_30/bias/vbatchnorm_31/gamma/vbatchnorm_31/beta/vconv_transpose_a/kernel/vconv_transpose_a/bias/vbatchnorm_b/gamma/vbatchnorm_b/beta/vconv_transpose_4a/kernel/vconv_transpose_4a/bias/v*?
Tin}
{2y*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_restore_5764??
?
?
F__inference_batchnorm_4a_layer_call_and_return_conditional_losses_1706

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_2553
input_layer!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:$

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:$

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:$

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:$

unknown_47:

unknown_48:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_2450y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameinput_layer
?
?
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_1545

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_leaky_relu_1_layer_call_fn_4234

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_1_layer_call_and_return_conditional_losses_2262j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?K
 __inference__traced_restore_5764
file_prefix8
assignvariableop_conv_1_kernel:,
assignvariableop_1_conv_1_bias:2
$assignvariableop_2_batchnorm_1_gamma:1
#assignvariableop_3_batchnorm_1_beta:8
*assignvariableop_4_batchnorm_1_moving_mean:<
.assignvariableop_5_batchnorm_1_moving_variance::
 assignvariableop_6_conv_2_kernel:,
assignvariableop_7_conv_2_bias:2
$assignvariableop_8_batchnorm_2_gamma:1
#assignvariableop_9_batchnorm_2_beta:9
+assignvariableop_10_batchnorm_2_moving_mean:=
/assignvariableop_11_batchnorm_2_moving_variance:;
!assignvariableop_12_conv_3_kernel:-
assignvariableop_13_conv_3_bias:3
%assignvariableop_14_batchnorm_3_gamma:2
$assignvariableop_15_batchnorm_3_beta:9
+assignvariableop_16_batchnorm_3_moving_mean:=
/assignvariableop_17_batchnorm_3_moving_variance:<
"assignvariableop_18_conv_4a_kernel:.
 assignvariableop_19_conv_4a_bias:4
&assignvariableop_20_batchnorm_4a_gamma:3
%assignvariableop_21_batchnorm_4a_beta::
,assignvariableop_22_batchnorm_4a_moving_mean:>
0assignvariableop_23_batchnorm_4a_moving_variance:E
+assignvariableop_24_conv_transpose_1_kernel:7
)assignvariableop_25_conv_transpose_1_bias:3
%assignvariableop_26_batchnorm_4_gamma:2
$assignvariableop_27_batchnorm_4_beta:9
+assignvariableop_28_batchnorm_4_moving_mean:=
/assignvariableop_29_batchnorm_4_moving_variance:E
+assignvariableop_30_conv_transpose_2_kernel:7
)assignvariableop_31_conv_transpose_2_bias:3
%assignvariableop_32_batchnorm_5_gamma:2
$assignvariableop_33_batchnorm_5_beta:9
+assignvariableop_34_batchnorm_5_moving_mean:=
/assignvariableop_35_batchnorm_5_moving_variance:F
,assignvariableop_36_conv_transpose_30_kernel:8
*assignvariableop_37_conv_transpose_30_bias:4
&assignvariableop_38_batchnorm_31_gamma:3
%assignvariableop_39_batchnorm_31_beta::
,assignvariableop_40_batchnorm_31_moving_mean:>
0assignvariableop_41_batchnorm_31_moving_variance:E
+assignvariableop_42_conv_transpose_a_kernel:7
)assignvariableop_43_conv_transpose_a_bias:3
%assignvariableop_44_batchnorm_b_gamma:2
$assignvariableop_45_batchnorm_b_beta:9
+assignvariableop_46_batchnorm_b_moving_mean:=
/assignvariableop_47_batchnorm_b_moving_variance:F
,assignvariableop_48_conv_transpose_4a_kernel:8
*assignvariableop_49_conv_transpose_4a_bias:#
assignvariableop_50_total: #
assignvariableop_51_count: =
#assignvariableop_52_conv_1_kernel_m:/
!assignvariableop_53_conv_1_bias_m:5
'assignvariableop_54_batchnorm_1_gamma_m:4
&assignvariableop_55_batchnorm_1_beta_m:=
#assignvariableop_56_conv_2_kernel_m:/
!assignvariableop_57_conv_2_bias_m:5
'assignvariableop_58_batchnorm_2_gamma_m:4
&assignvariableop_59_batchnorm_2_beta_m:=
#assignvariableop_60_conv_3_kernel_m:/
!assignvariableop_61_conv_3_bias_m:5
'assignvariableop_62_batchnorm_3_gamma_m:4
&assignvariableop_63_batchnorm_3_beta_m:>
$assignvariableop_64_conv_4a_kernel_m:0
"assignvariableop_65_conv_4a_bias_m:6
(assignvariableop_66_batchnorm_4a_gamma_m:5
'assignvariableop_67_batchnorm_4a_beta_m:G
-assignvariableop_68_conv_transpose_1_kernel_m:9
+assignvariableop_69_conv_transpose_1_bias_m:5
'assignvariableop_70_batchnorm_4_gamma_m:4
&assignvariableop_71_batchnorm_4_beta_m:G
-assignvariableop_72_conv_transpose_2_kernel_m:9
+assignvariableop_73_conv_transpose_2_bias_m:5
'assignvariableop_74_batchnorm_5_gamma_m:4
&assignvariableop_75_batchnorm_5_beta_m:H
.assignvariableop_76_conv_transpose_30_kernel_m::
,assignvariableop_77_conv_transpose_30_bias_m:6
(assignvariableop_78_batchnorm_31_gamma_m:5
'assignvariableop_79_batchnorm_31_beta_m:G
-assignvariableop_80_conv_transpose_a_kernel_m:9
+assignvariableop_81_conv_transpose_a_bias_m:5
'assignvariableop_82_batchnorm_b_gamma_m:4
&assignvariableop_83_batchnorm_b_beta_m:H
.assignvariableop_84_conv_transpose_4a_kernel_m::
,assignvariableop_85_conv_transpose_4a_bias_m:=
#assignvariableop_86_conv_1_kernel_v:/
!assignvariableop_87_conv_1_bias_v:5
'assignvariableop_88_batchnorm_1_gamma_v:4
&assignvariableop_89_batchnorm_1_beta_v:=
#assignvariableop_90_conv_2_kernel_v:/
!assignvariableop_91_conv_2_bias_v:5
'assignvariableop_92_batchnorm_2_gamma_v:4
&assignvariableop_93_batchnorm_2_beta_v:=
#assignvariableop_94_conv_3_kernel_v:/
!assignvariableop_95_conv_3_bias_v:5
'assignvariableop_96_batchnorm_3_gamma_v:4
&assignvariableop_97_batchnorm_3_beta_v:>
$assignvariableop_98_conv_4a_kernel_v:0
"assignvariableop_99_conv_4a_bias_v:7
)assignvariableop_100_batchnorm_4a_gamma_v:6
(assignvariableop_101_batchnorm_4a_beta_v:H
.assignvariableop_102_conv_transpose_1_kernel_v::
,assignvariableop_103_conv_transpose_1_bias_v:6
(assignvariableop_104_batchnorm_4_gamma_v:5
'assignvariableop_105_batchnorm_4_beta_v:H
.assignvariableop_106_conv_transpose_2_kernel_v::
,assignvariableop_107_conv_transpose_2_bias_v:6
(assignvariableop_108_batchnorm_5_gamma_v:5
'assignvariableop_109_batchnorm_5_beta_v:I
/assignvariableop_110_conv_transpose_30_kernel_v:;
-assignvariableop_111_conv_transpose_30_bias_v:7
)assignvariableop_112_batchnorm_31_gamma_v:6
(assignvariableop_113_batchnorm_31_beta_v:H
.assignvariableop_114_conv_transpose_a_kernel_v::
,assignvariableop_115_conv_transpose_a_bias_v:6
(assignvariableop_116_batchnorm_b_gamma_v:5
'assignvariableop_117_batchnorm_b_beta_v:I
/assignvariableop_118_conv_transpose_4a_kernel_v:;
-assignvariableop_119_conv_transpose_4a_bias_v:
identity_121??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?D
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:y*
dtype0*?D
value?CB?CyB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:y*
dtype0*?
value?B?yB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes}
{2y[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_conv_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp$assignvariableop_2_batchnorm_1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp#assignvariableop_3_batchnorm_1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp*assignvariableop_4_batchnorm_1_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batchnorm_1_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp$assignvariableop_8_batchnorm_2_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_batchnorm_2_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp+assignvariableop_10_batchnorm_2_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batchnorm_2_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_conv_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp%assignvariableop_14_batchnorm_3_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp$assignvariableop_15_batchnorm_3_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp+assignvariableop_16_batchnorm_3_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batchnorm_3_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp"assignvariableop_18_conv_4a_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp assignvariableop_19_conv_4a_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_batchnorm_4a_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp%assignvariableop_21_batchnorm_4a_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp,assignvariableop_22_batchnorm_4a_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp0assignvariableop_23_batchnorm_4a_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp+assignvariableop_24_conv_transpose_1_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp)assignvariableop_25_conv_transpose_1_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp%assignvariableop_26_batchnorm_4_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp$assignvariableop_27_batchnorm_4_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_batchnorm_4_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp/assignvariableop_29_batchnorm_4_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp+assignvariableop_30_conv_transpose_2_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_conv_transpose_2_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp%assignvariableop_32_batchnorm_5_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp$assignvariableop_33_batchnorm_5_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp+assignvariableop_34_batchnorm_5_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp/assignvariableop_35_batchnorm_5_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp,assignvariableop_36_conv_transpose_30_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_conv_transpose_30_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp&assignvariableop_38_batchnorm_31_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp%assignvariableop_39_batchnorm_31_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp,assignvariableop_40_batchnorm_31_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp0assignvariableop_41_batchnorm_31_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp+assignvariableop_42_conv_transpose_a_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp)assignvariableop_43_conv_transpose_a_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp%assignvariableop_44_batchnorm_b_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp$assignvariableop_45_batchnorm_b_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp+assignvariableop_46_batchnorm_b_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp/assignvariableop_47_batchnorm_b_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp,assignvariableop_48_conv_transpose_4a_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp*assignvariableop_49_conv_transpose_4a_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOpassignvariableop_50_totalIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOpassignvariableop_51_countIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp#assignvariableop_52_conv_1_kernel_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp!assignvariableop_53_conv_1_bias_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp'assignvariableop_54_batchnorm_1_gamma_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp&assignvariableop_55_batchnorm_1_beta_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp#assignvariableop_56_conv_2_kernel_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp!assignvariableop_57_conv_2_bias_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp'assignvariableop_58_batchnorm_2_gamma_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp&assignvariableop_59_batchnorm_2_beta_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp#assignvariableop_60_conv_3_kernel_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp!assignvariableop_61_conv_3_bias_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp'assignvariableop_62_batchnorm_3_gamma_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp&assignvariableop_63_batchnorm_3_beta_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp$assignvariableop_64_conv_4a_kernel_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp"assignvariableop_65_conv_4a_bias_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp(assignvariableop_66_batchnorm_4a_gamma_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp'assignvariableop_67_batchnorm_4a_beta_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp-assignvariableop_68_conv_transpose_1_kernel_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp+assignvariableop_69_conv_transpose_1_bias_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp'assignvariableop_70_batchnorm_4_gamma_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp&assignvariableop_71_batchnorm_4_beta_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp-assignvariableop_72_conv_transpose_2_kernel_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp+assignvariableop_73_conv_transpose_2_bias_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp'assignvariableop_74_batchnorm_5_gamma_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp&assignvariableop_75_batchnorm_5_beta_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOp.assignvariableop_76_conv_transpose_30_kernel_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOp,assignvariableop_77_conv_transpose_30_bias_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp(assignvariableop_78_batchnorm_31_gamma_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp'assignvariableop_79_batchnorm_31_beta_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOp-assignvariableop_80_conv_transpose_a_kernel_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOp+assignvariableop_81_conv_transpose_a_bias_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOp'assignvariableop_82_batchnorm_b_gamma_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOp&assignvariableop_83_batchnorm_b_beta_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOp.assignvariableop_84_conv_transpose_4a_kernel_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOp,assignvariableop_85_conv_transpose_4a_bias_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOp#assignvariableop_86_conv_1_kernel_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_87AssignVariableOp!assignvariableop_87_conv_1_bias_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_88AssignVariableOp'assignvariableop_88_batchnorm_1_gamma_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_89AssignVariableOp&assignvariableop_89_batchnorm_1_beta_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_90AssignVariableOp#assignvariableop_90_conv_2_kernel_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_91AssignVariableOp!assignvariableop_91_conv_2_bias_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_92AssignVariableOp'assignvariableop_92_batchnorm_2_gamma_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_93AssignVariableOp&assignvariableop_93_batchnorm_2_beta_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_94AssignVariableOp#assignvariableop_94_conv_3_kernel_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_95AssignVariableOp!assignvariableop_95_conv_3_bias_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_96AssignVariableOp'assignvariableop_96_batchnorm_3_gamma_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_97AssignVariableOp&assignvariableop_97_batchnorm_3_beta_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_98AssignVariableOp$assignvariableop_98_conv_4a_kernel_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_99AssignVariableOp"assignvariableop_99_conv_4a_bias_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_100AssignVariableOp)assignvariableop_100_batchnorm_4a_gamma_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_101AssignVariableOp(assignvariableop_101_batchnorm_4a_beta_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_102AssignVariableOp.assignvariableop_102_conv_transpose_1_kernel_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_103AssignVariableOp,assignvariableop_103_conv_transpose_1_bias_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_104AssignVariableOp(assignvariableop_104_batchnorm_4_gamma_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_105AssignVariableOp'assignvariableop_105_batchnorm_4_beta_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_106AssignVariableOp.assignvariableop_106_conv_transpose_2_kernel_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_107AssignVariableOp,assignvariableop_107_conv_transpose_2_bias_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_108AssignVariableOp(assignvariableop_108_batchnorm_5_gamma_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_109AssignVariableOp'assignvariableop_109_batchnorm_5_beta_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_110AssignVariableOp/assignvariableop_110_conv_transpose_30_kernel_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_111AssignVariableOp-assignvariableop_111_conv_transpose_30_bias_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_112AssignVariableOp)assignvariableop_112_batchnorm_31_gamma_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_113AssignVariableOp(assignvariableop_113_batchnorm_31_beta_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_114AssignVariableOp.assignvariableop_114_conv_transpose_a_kernel_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_115AssignVariableOp,assignvariableop_115_conv_transpose_a_bias_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_116AssignVariableOp(assignvariableop_116_batchnorm_b_gamma_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_117AssignVariableOp'assignvariableop_117_batchnorm_b_beta_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_118AssignVariableOp/assignvariableop_118_conv_transpose_4a_kernel_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_119AssignVariableOp-assignvariableop_119_conv_transpose_4a_bias_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_120Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_121IdentityIdentity_120:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_121Identity_121:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
E__inference_batchnorm_b_layer_call_and_return_conditional_losses_2138

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_31_layer_call_and_return_conditional_losses_4826

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?x
?
?__inference_model_layer_call_and_return_conditional_losses_3343
input_layer%
conv_1_3217:
conv_1_3219:
batchnorm_1_3222:
batchnorm_1_3224:
batchnorm_1_3226:
batchnorm_1_3228:%
conv_2_3232:
conv_2_3234:
batchnorm_2_3237:
batchnorm_2_3239:
batchnorm_2_3241:
batchnorm_2_3243:%
conv_3_3247:
conv_3_3249:
batchnorm_3_3252:
batchnorm_3_3254:
batchnorm_3_3256:
batchnorm_3_3258:&
conv_4a_3262:
conv_4a_3264:
batchnorm_4a_3267:
batchnorm_4a_3269:
batchnorm_4a_3271:
batchnorm_4a_3273:/
conv_transpose_1_3277:#
conv_transpose_1_3279:
batchnorm_4_3282:
batchnorm_4_3284:
batchnorm_4_3286:
batchnorm_4_3288:/
conv_transpose_2_3292:#
conv_transpose_2_3294:
batchnorm_5_3297:
batchnorm_5_3299:
batchnorm_5_3301:
batchnorm_5_3303:0
conv_transpose_30_3307:$
conv_transpose_30_3309:
batchnorm_31_3312:
batchnorm_31_3314:
batchnorm_31_3316:
batchnorm_31_3318:/
conv_transpose_a_3322:#
conv_transpose_a_3324:
batchnorm_b_3327:
batchnorm_b_3329:
batchnorm_b_3331:
batchnorm_b_3333:0
conv_transpose_4a_3337:$
conv_transpose_4a_3339:
identity??#batchnorm_1/StatefulPartitionedCall?#batchnorm_2/StatefulPartitionedCall?#batchnorm_3/StatefulPartitionedCall?$batchnorm_31/StatefulPartitionedCall?#batchnorm_4/StatefulPartitionedCall?$batchnorm_4a/StatefulPartitionedCall?#batchnorm_5/StatefulPartitionedCall?#batchnorm_b/StatefulPartitionedCall?conv_1/StatefulPartitionedCall?conv_2/StatefulPartitionedCall?conv_3/StatefulPartitionedCall?conv_4a/StatefulPartitionedCall?(conv_transpose_1/StatefulPartitionedCall?(conv_transpose_2/StatefulPartitionedCall?)conv_transpose_30/StatefulPartitionedCall?)conv_transpose_4a/StatefulPartitionedCall?(conv_transpose_a/StatefulPartitionedCall?
conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv_1_3217conv_1_3219*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_1_layer_call_and_return_conditional_losses_2242?
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batchnorm_1_3222batchnorm_1_3224batchnorm_1_3226batchnorm_1_3228*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_1545?
leaky_relu_1/PartitionedCallPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_1_layer_call_and_return_conditional_losses_2262?
conv_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_1/PartitionedCall:output:0conv_2_3232conv_2_3234*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_2_layer_call_and_return_conditional_losses_2274?
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batchnorm_2_3237batchnorm_2_3239batchnorm_2_3241batchnorm_2_3243*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_1609?
leaky_relu_2/PartitionedCallPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_2_layer_call_and_return_conditional_losses_2294?
conv_3/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_2/PartitionedCall:output:0conv_3_3247conv_3_3249*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_3_layer_call_and_return_conditional_losses_2306?
#batchnorm_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batchnorm_3_3252batchnorm_3_3254batchnorm_3_3256batchnorm_3_3258*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_1673?
leaky_relu_3/PartitionedCallPartitionedCall,batchnorm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_3_layer_call_and_return_conditional_losses_2326?
conv_4a/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_3/PartitionedCall:output:0conv_4a_3262conv_4a_3264*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv_4a_layer_call_and_return_conditional_losses_2338?
$batchnorm_4a/StatefulPartitionedCallStatefulPartitionedCall(conv_4a/StatefulPartitionedCall:output:0batchnorm_4a_3267batchnorm_4a_3269batchnorm_4a_3271batchnorm_4a_3273*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_batchnorm_4a_layer_call_and_return_conditional_losses_1737?
leaky_relu_4a/PartitionedCallPartitionedCall-batchnorm_4a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_leaky_relu_4a_layer_call_and_return_conditional_losses_2358?
(conv_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_relu_4a/PartitionedCall:output:0conv_transpose_1_3277conv_transpose_1_3279*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_1_layer_call_and_return_conditional_losses_1785?
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_1/StatefulPartitionedCall:output:0batchnorm_4_3282batchnorm_4_3284batchnorm_4_3286batchnorm_4_3288*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_1845?
leaky_relu_4/PartitionedCallPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_4_layer_call_and_return_conditional_losses_2379?
(conv_transpose_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_4/PartitionedCall:output:0conv_transpose_2_3292conv_transpose_2_3294*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_2_layer_call_and_return_conditional_losses_1893?
#batchnorm_5/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_2/StatefulPartitionedCall:output:0batchnorm_5_3297batchnorm_5_3299batchnorm_5_3301batchnorm_5_3303*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_1953?
leaky_relu_5/PartitionedCallPartitionedCall,batchnorm_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_5_layer_call_and_return_conditional_losses_2400?
)conv_transpose_30/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_5/PartitionedCall:output:0conv_transpose_30_3307conv_transpose_30_3309*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv_transpose_30_layer_call_and_return_conditional_losses_2001?
$batchnorm_31/StatefulPartitionedCallStatefulPartitionedCall2conv_transpose_30/StatefulPartitionedCall:output:0batchnorm_31_3312batchnorm_31_3314batchnorm_31_3316batchnorm_31_3318*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_batchnorm_31_layer_call_and_return_conditional_losses_2061?
leaky_relu_32/PartitionedCallPartitionedCall-batchnorm_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_leaky_relu_32_layer_call_and_return_conditional_losses_2421?
(conv_transpose_a/StatefulPartitionedCallStatefulPartitionedCall&leaky_relu_32/PartitionedCall:output:0conv_transpose_a_3322conv_transpose_a_3324*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_a_layer_call_and_return_conditional_losses_2109?
#batchnorm_b/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_a/StatefulPartitionedCall:output:0batchnorm_b_3327batchnorm_b_3329batchnorm_b_3331batchnorm_b_3333*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_b_layer_call_and_return_conditional_losses_2169?
leaky_relu_c/PartitionedCallPartitionedCall,batchnorm_b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_c_layer_call_and_return_conditional_losses_2442?
)conv_transpose_4a/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_c/PartitionedCall:output:0conv_transpose_4a_3337conv_transpose_4a_3339*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv_transpose_4a_layer_call_and_return_conditional_losses_2218?
IdentityIdentity2conv_transpose_4a/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_3/StatefulPartitionedCall%^batchnorm_31/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall%^batchnorm_4a/StatefulPartitionedCall$^batchnorm_5/StatefulPartitionedCall$^batchnorm_b/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall ^conv_4a/StatefulPartitionedCall)^conv_transpose_1/StatefulPartitionedCall)^conv_transpose_2/StatefulPartitionedCall*^conv_transpose_30/StatefulPartitionedCall*^conv_transpose_4a/StatefulPartitionedCall)^conv_transpose_a/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_3/StatefulPartitionedCall#batchnorm_3/StatefulPartitionedCall2L
$batchnorm_31/StatefulPartitionedCall$batchnorm_31/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2L
$batchnorm_4a/StatefulPartitionedCall$batchnorm_4a/StatefulPartitionedCall2J
#batchnorm_5/StatefulPartitionedCall#batchnorm_5/StatefulPartitionedCall2J
#batchnorm_b/StatefulPartitionedCall#batchnorm_b/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2B
conv_4a/StatefulPartitionedCallconv_4a/StatefulPartitionedCall2T
(conv_transpose_1/StatefulPartitionedCall(conv_transpose_1/StatefulPartitionedCall2T
(conv_transpose_2/StatefulPartitionedCall(conv_transpose_2/StatefulPartitionedCall2V
)conv_transpose_30/StatefulPartitionedCall)conv_transpose_30/StatefulPartitionedCall2V
)conv_transpose_4a/StatefulPartitionedCall)conv_transpose_4a/StatefulPartitionedCall2T
(conv_transpose_a/StatefulPartitionedCall(conv_transpose_a/StatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameinput_layer
?
b
F__inference_leaky_relu_5_layer_call_and_return_conditional_losses_2400

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_leaky_relu_4a_layer_call_and_return_conditional_losses_2358

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_leaky_relu_4_layer_call_and_return_conditional_losses_2379

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_1953

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_2_layer_call_fn_4284

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_1609?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
/__inference_conv_transpose_a_layer_call_fn_4863

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_a_layer_call_and_return_conditional_losses_2109?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_4a_layer_call_fn_4466

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_batchnorm_4a_layer_call_and_return_conditional_losses_1737?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_b_layer_call_and_return_conditional_losses_4958

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_leaky_relu_4_layer_call_fn_4621

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_4_layer_call_and_return_conditional_losses_2379h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_1514

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_leaky_relu_2_layer_call_and_return_conditional_losses_4330

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????  *
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?x
?
?__inference_model_layer_call_and_return_conditional_losses_2877

inputs%
conv_1_2751:
conv_1_2753:
batchnorm_1_2756:
batchnorm_1_2758:
batchnorm_1_2760:
batchnorm_1_2762:%
conv_2_2766:
conv_2_2768:
batchnorm_2_2771:
batchnorm_2_2773:
batchnorm_2_2775:
batchnorm_2_2777:%
conv_3_2781:
conv_3_2783:
batchnorm_3_2786:
batchnorm_3_2788:
batchnorm_3_2790:
batchnorm_3_2792:&
conv_4a_2796:
conv_4a_2798:
batchnorm_4a_2801:
batchnorm_4a_2803:
batchnorm_4a_2805:
batchnorm_4a_2807:/
conv_transpose_1_2811:#
conv_transpose_1_2813:
batchnorm_4_2816:
batchnorm_4_2818:
batchnorm_4_2820:
batchnorm_4_2822:/
conv_transpose_2_2826:#
conv_transpose_2_2828:
batchnorm_5_2831:
batchnorm_5_2833:
batchnorm_5_2835:
batchnorm_5_2837:0
conv_transpose_30_2841:$
conv_transpose_30_2843:
batchnorm_31_2846:
batchnorm_31_2848:
batchnorm_31_2850:
batchnorm_31_2852:/
conv_transpose_a_2856:#
conv_transpose_a_2858:
batchnorm_b_2861:
batchnorm_b_2863:
batchnorm_b_2865:
batchnorm_b_2867:0
conv_transpose_4a_2871:$
conv_transpose_4a_2873:
identity??#batchnorm_1/StatefulPartitionedCall?#batchnorm_2/StatefulPartitionedCall?#batchnorm_3/StatefulPartitionedCall?$batchnorm_31/StatefulPartitionedCall?#batchnorm_4/StatefulPartitionedCall?$batchnorm_4a/StatefulPartitionedCall?#batchnorm_5/StatefulPartitionedCall?#batchnorm_b/StatefulPartitionedCall?conv_1/StatefulPartitionedCall?conv_2/StatefulPartitionedCall?conv_3/StatefulPartitionedCall?conv_4a/StatefulPartitionedCall?(conv_transpose_1/StatefulPartitionedCall?(conv_transpose_2/StatefulPartitionedCall?)conv_transpose_30/StatefulPartitionedCall?)conv_transpose_4a/StatefulPartitionedCall?(conv_transpose_a/StatefulPartitionedCall?
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_2751conv_1_2753*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_1_layer_call_and_return_conditional_losses_2242?
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batchnorm_1_2756batchnorm_1_2758batchnorm_1_2760batchnorm_1_2762*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_1545?
leaky_relu_1/PartitionedCallPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_1_layer_call_and_return_conditional_losses_2262?
conv_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_1/PartitionedCall:output:0conv_2_2766conv_2_2768*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_2_layer_call_and_return_conditional_losses_2274?
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batchnorm_2_2771batchnorm_2_2773batchnorm_2_2775batchnorm_2_2777*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_1609?
leaky_relu_2/PartitionedCallPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_2_layer_call_and_return_conditional_losses_2294?
conv_3/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_2/PartitionedCall:output:0conv_3_2781conv_3_2783*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_3_layer_call_and_return_conditional_losses_2306?
#batchnorm_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batchnorm_3_2786batchnorm_3_2788batchnorm_3_2790batchnorm_3_2792*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_1673?
leaky_relu_3/PartitionedCallPartitionedCall,batchnorm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_3_layer_call_and_return_conditional_losses_2326?
conv_4a/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_3/PartitionedCall:output:0conv_4a_2796conv_4a_2798*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv_4a_layer_call_and_return_conditional_losses_2338?
$batchnorm_4a/StatefulPartitionedCallStatefulPartitionedCall(conv_4a/StatefulPartitionedCall:output:0batchnorm_4a_2801batchnorm_4a_2803batchnorm_4a_2805batchnorm_4a_2807*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_batchnorm_4a_layer_call_and_return_conditional_losses_1737?
leaky_relu_4a/PartitionedCallPartitionedCall-batchnorm_4a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_leaky_relu_4a_layer_call_and_return_conditional_losses_2358?
(conv_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_relu_4a/PartitionedCall:output:0conv_transpose_1_2811conv_transpose_1_2813*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_1_layer_call_and_return_conditional_losses_1785?
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_1/StatefulPartitionedCall:output:0batchnorm_4_2816batchnorm_4_2818batchnorm_4_2820batchnorm_4_2822*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_1845?
leaky_relu_4/PartitionedCallPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_4_layer_call_and_return_conditional_losses_2379?
(conv_transpose_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_4/PartitionedCall:output:0conv_transpose_2_2826conv_transpose_2_2828*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_2_layer_call_and_return_conditional_losses_1893?
#batchnorm_5/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_2/StatefulPartitionedCall:output:0batchnorm_5_2831batchnorm_5_2833batchnorm_5_2835batchnorm_5_2837*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_1953?
leaky_relu_5/PartitionedCallPartitionedCall,batchnorm_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_5_layer_call_and_return_conditional_losses_2400?
)conv_transpose_30/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_5/PartitionedCall:output:0conv_transpose_30_2841conv_transpose_30_2843*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv_transpose_30_layer_call_and_return_conditional_losses_2001?
$batchnorm_31/StatefulPartitionedCallStatefulPartitionedCall2conv_transpose_30/StatefulPartitionedCall:output:0batchnorm_31_2846batchnorm_31_2848batchnorm_31_2850batchnorm_31_2852*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_batchnorm_31_layer_call_and_return_conditional_losses_2061?
leaky_relu_32/PartitionedCallPartitionedCall-batchnorm_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_leaky_relu_32_layer_call_and_return_conditional_losses_2421?
(conv_transpose_a/StatefulPartitionedCallStatefulPartitionedCall&leaky_relu_32/PartitionedCall:output:0conv_transpose_a_2856conv_transpose_a_2858*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_a_layer_call_and_return_conditional_losses_2109?
#batchnorm_b/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_a/StatefulPartitionedCall:output:0batchnorm_b_2861batchnorm_b_2863batchnorm_b_2865batchnorm_b_2867*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_b_layer_call_and_return_conditional_losses_2169?
leaky_relu_c/PartitionedCallPartitionedCall,batchnorm_b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_c_layer_call_and_return_conditional_losses_2442?
)conv_transpose_4a/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_c/PartitionedCall:output:0conv_transpose_4a_2871conv_transpose_4a_2873*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv_transpose_4a_layer_call_and_return_conditional_losses_2218?
IdentityIdentity2conv_transpose_4a/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_3/StatefulPartitionedCall%^batchnorm_31/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall%^batchnorm_4a/StatefulPartitionedCall$^batchnorm_5/StatefulPartitionedCall$^batchnorm_b/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall ^conv_4a/StatefulPartitionedCall)^conv_transpose_1/StatefulPartitionedCall)^conv_transpose_2/StatefulPartitionedCall*^conv_transpose_30/StatefulPartitionedCall*^conv_transpose_4a/StatefulPartitionedCall)^conv_transpose_a/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_3/StatefulPartitionedCall#batchnorm_3/StatefulPartitionedCall2L
$batchnorm_31/StatefulPartitionedCall$batchnorm_31/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2L
$batchnorm_4a/StatefulPartitionedCall$batchnorm_4a/StatefulPartitionedCall2J
#batchnorm_5/StatefulPartitionedCall#batchnorm_5/StatefulPartitionedCall2J
#batchnorm_b/StatefulPartitionedCall#batchnorm_b/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2B
conv_4a/StatefulPartitionedCallconv_4a/StatefulPartitionedCall2T
(conv_transpose_1/StatefulPartitionedCall(conv_transpose_1/StatefulPartitionedCall2T
(conv_transpose_2/StatefulPartitionedCall(conv_transpose_2/StatefulPartitionedCall2V
)conv_transpose_30/StatefulPartitionedCall)conv_transpose_30/StatefulPartitionedCall2V
)conv_transpose_4a/StatefulPartitionedCall)conv_transpose_4a/StatefulPartitionedCall2T
(conv_transpose_a/StatefulPartitionedCall(conv_transpose_a/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_31_layer_call_and_return_conditional_losses_2030

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
%__inference_conv_3_layer_call_fn_4339

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_3_layer_call_and_return_conditional_losses_2306w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?x
?
?__inference_model_layer_call_and_return_conditional_losses_2450

inputs%
conv_1_2243:
conv_1_2245:
batchnorm_1_2248:
batchnorm_1_2250:
batchnorm_1_2252:
batchnorm_1_2254:%
conv_2_2275:
conv_2_2277:
batchnorm_2_2280:
batchnorm_2_2282:
batchnorm_2_2284:
batchnorm_2_2286:%
conv_3_2307:
conv_3_2309:
batchnorm_3_2312:
batchnorm_3_2314:
batchnorm_3_2316:
batchnorm_3_2318:&
conv_4a_2339:
conv_4a_2341:
batchnorm_4a_2344:
batchnorm_4a_2346:
batchnorm_4a_2348:
batchnorm_4a_2350:/
conv_transpose_1_2360:#
conv_transpose_1_2362:
batchnorm_4_2365:
batchnorm_4_2367:
batchnorm_4_2369:
batchnorm_4_2371:/
conv_transpose_2_2381:#
conv_transpose_2_2383:
batchnorm_5_2386:
batchnorm_5_2388:
batchnorm_5_2390:
batchnorm_5_2392:0
conv_transpose_30_2402:$
conv_transpose_30_2404:
batchnorm_31_2407:
batchnorm_31_2409:
batchnorm_31_2411:
batchnorm_31_2413:/
conv_transpose_a_2423:#
conv_transpose_a_2425:
batchnorm_b_2428:
batchnorm_b_2430:
batchnorm_b_2432:
batchnorm_b_2434:0
conv_transpose_4a_2444:$
conv_transpose_4a_2446:
identity??#batchnorm_1/StatefulPartitionedCall?#batchnorm_2/StatefulPartitionedCall?#batchnorm_3/StatefulPartitionedCall?$batchnorm_31/StatefulPartitionedCall?#batchnorm_4/StatefulPartitionedCall?$batchnorm_4a/StatefulPartitionedCall?#batchnorm_5/StatefulPartitionedCall?#batchnorm_b/StatefulPartitionedCall?conv_1/StatefulPartitionedCall?conv_2/StatefulPartitionedCall?conv_3/StatefulPartitionedCall?conv_4a/StatefulPartitionedCall?(conv_transpose_1/StatefulPartitionedCall?(conv_transpose_2/StatefulPartitionedCall?)conv_transpose_30/StatefulPartitionedCall?)conv_transpose_4a/StatefulPartitionedCall?(conv_transpose_a/StatefulPartitionedCall?
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_2243conv_1_2245*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_1_layer_call_and_return_conditional_losses_2242?
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batchnorm_1_2248batchnorm_1_2250batchnorm_1_2252batchnorm_1_2254*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_1514?
leaky_relu_1/PartitionedCallPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_1_layer_call_and_return_conditional_losses_2262?
conv_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_1/PartitionedCall:output:0conv_2_2275conv_2_2277*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_2_layer_call_and_return_conditional_losses_2274?
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batchnorm_2_2280batchnorm_2_2282batchnorm_2_2284batchnorm_2_2286*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_1578?
leaky_relu_2/PartitionedCallPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_2_layer_call_and_return_conditional_losses_2294?
conv_3/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_2/PartitionedCall:output:0conv_3_2307conv_3_2309*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_3_layer_call_and_return_conditional_losses_2306?
#batchnorm_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batchnorm_3_2312batchnorm_3_2314batchnorm_3_2316batchnorm_3_2318*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_1642?
leaky_relu_3/PartitionedCallPartitionedCall,batchnorm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_3_layer_call_and_return_conditional_losses_2326?
conv_4a/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_3/PartitionedCall:output:0conv_4a_2339conv_4a_2341*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv_4a_layer_call_and_return_conditional_losses_2338?
$batchnorm_4a/StatefulPartitionedCallStatefulPartitionedCall(conv_4a/StatefulPartitionedCall:output:0batchnorm_4a_2344batchnorm_4a_2346batchnorm_4a_2348batchnorm_4a_2350*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_batchnorm_4a_layer_call_and_return_conditional_losses_1706?
leaky_relu_4a/PartitionedCallPartitionedCall-batchnorm_4a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_leaky_relu_4a_layer_call_and_return_conditional_losses_2358?
(conv_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_relu_4a/PartitionedCall:output:0conv_transpose_1_2360conv_transpose_1_2362*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_1_layer_call_and_return_conditional_losses_1785?
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_1/StatefulPartitionedCall:output:0batchnorm_4_2365batchnorm_4_2367batchnorm_4_2369batchnorm_4_2371*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_1814?
leaky_relu_4/PartitionedCallPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_4_layer_call_and_return_conditional_losses_2379?
(conv_transpose_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_4/PartitionedCall:output:0conv_transpose_2_2381conv_transpose_2_2383*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_2_layer_call_and_return_conditional_losses_1893?
#batchnorm_5/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_2/StatefulPartitionedCall:output:0batchnorm_5_2386batchnorm_5_2388batchnorm_5_2390batchnorm_5_2392*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_1922?
leaky_relu_5/PartitionedCallPartitionedCall,batchnorm_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_5_layer_call_and_return_conditional_losses_2400?
)conv_transpose_30/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_5/PartitionedCall:output:0conv_transpose_30_2402conv_transpose_30_2404*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv_transpose_30_layer_call_and_return_conditional_losses_2001?
$batchnorm_31/StatefulPartitionedCallStatefulPartitionedCall2conv_transpose_30/StatefulPartitionedCall:output:0batchnorm_31_2407batchnorm_31_2409batchnorm_31_2411batchnorm_31_2413*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_batchnorm_31_layer_call_and_return_conditional_losses_2030?
leaky_relu_32/PartitionedCallPartitionedCall-batchnorm_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_leaky_relu_32_layer_call_and_return_conditional_losses_2421?
(conv_transpose_a/StatefulPartitionedCallStatefulPartitionedCall&leaky_relu_32/PartitionedCall:output:0conv_transpose_a_2423conv_transpose_a_2425*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_a_layer_call_and_return_conditional_losses_2109?
#batchnorm_b/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_a/StatefulPartitionedCall:output:0batchnorm_b_2428batchnorm_b_2430batchnorm_b_2432batchnorm_b_2434*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_b_layer_call_and_return_conditional_losses_2138?
leaky_relu_c/PartitionedCallPartitionedCall,batchnorm_b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_c_layer_call_and_return_conditional_losses_2442?
)conv_transpose_4a/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_c/PartitionedCall:output:0conv_transpose_4a_2444conv_transpose_4a_2446*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv_transpose_4a_layer_call_and_return_conditional_losses_2218?
IdentityIdentity2conv_transpose_4a/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_3/StatefulPartitionedCall%^batchnorm_31/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall%^batchnorm_4a/StatefulPartitionedCall$^batchnorm_5/StatefulPartitionedCall$^batchnorm_b/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall ^conv_4a/StatefulPartitionedCall)^conv_transpose_1/StatefulPartitionedCall)^conv_transpose_2/StatefulPartitionedCall*^conv_transpose_30/StatefulPartitionedCall*^conv_transpose_4a/StatefulPartitionedCall)^conv_transpose_a/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_3/StatefulPartitionedCall#batchnorm_3/StatefulPartitionedCall2L
$batchnorm_31/StatefulPartitionedCall$batchnorm_31/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2L
$batchnorm_4a/StatefulPartitionedCall$batchnorm_4a/StatefulPartitionedCall2J
#batchnorm_5/StatefulPartitionedCall#batchnorm_5/StatefulPartitionedCall2J
#batchnorm_b/StatefulPartitionedCall#batchnorm_b/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2B
conv_4a/StatefulPartitionedCallconv_4a/StatefulPartitionedCall2T
(conv_transpose_1/StatefulPartitionedCall(conv_transpose_1/StatefulPartitionedCall2T
(conv_transpose_2/StatefulPartitionedCall(conv_transpose_2/StatefulPartitionedCall2V
)conv_transpose_30/StatefulPartitionedCall)conv_transpose_30/StatefulPartitionedCall2V
)conv_transpose_4a/StatefulPartitionedCall)conv_transpose_4a/StatefulPartitionedCall2T
(conv_transpose_a/StatefulPartitionedCall(conv_transpose_a/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_leaky_relu_5_layer_call_and_return_conditional_losses_4740

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_conv_4a_layer_call_fn_4430

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv_4a_layer_call_and_return_conditional_losses_2338w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
?
K__inference_conv_transpose_30_layer_call_and_return_conditional_losses_4782

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
? 
?
J__inference_conv_transpose_a_layer_call_and_return_conditional_losses_4896

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_4a_layer_call_and_return_conditional_losses_1737

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
%__inference_conv_2_layer_call_fn_4248

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_2_layer_call_and_return_conditional_losses_2274w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
/__inference_conv_transpose_2_layer_call_fn_4635

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_2_layer_call_and_return_conditional_losses_1893?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_leaky_relu_2_layer_call_and_return_conditional_losses_2294

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????  *
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?!
?
K__inference_conv_transpose_4a_layer_call_and_return_conditional_losses_5011

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_4411

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_leaky_relu_c_layer_call_and_return_conditional_losses_4968

inputs
identitya
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:???????????*
alpha%???>i
IdentityIdentityLeakyRelu:activations:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_4_layer_call_fn_4580

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_1845?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_1845

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_leaky_relu_3_layer_call_fn_4416

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_3_layer_call_and_return_conditional_losses_2326h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_leaky_relu_3_layer_call_and_return_conditional_losses_2326

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_31_layer_call_and_return_conditional_losses_2061

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_leaky_relu_2_layer_call_fn_4325

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_2_layer_call_and_return_conditional_losses_2294h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_1673

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
A__inference_conv_4a_layer_call_and_return_conditional_losses_4440

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_3553

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:$

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:$

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:$

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:$

unknown_47:

unknown_48:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*D
_read_only_resource_inputs&
$"	
 !"%&'(+,-.12*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_2877y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
@__inference_conv_2_layer_call_and_return_conditional_losses_4258

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????  w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
G
+__inference_leaky_relu_c_layer_call_fn_4963

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_c_layer_call_and_return_conditional_losses_2442j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
G__inference_leaky_relu_4a_layer_call_and_return_conditional_losses_4512

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_b_layer_call_fn_4922

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_b_layer_call_and_return_conditional_losses_2169?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?,
?__inference_model_layer_call_and_return_conditional_losses_4041

inputs?
%conv_1_conv2d_readvariableop_resource:4
&conv_1_biasadd_readvariableop_resource:1
#batchnorm_1_readvariableop_resource:3
%batchnorm_1_readvariableop_1_resource:B
4batchnorm_1_fusedbatchnormv3_readvariableop_resource:D
6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource:?
%conv_2_conv2d_readvariableop_resource:4
&conv_2_biasadd_readvariableop_resource:1
#batchnorm_2_readvariableop_resource:3
%batchnorm_2_readvariableop_1_resource:B
4batchnorm_2_fusedbatchnormv3_readvariableop_resource:D
6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource:?
%conv_3_conv2d_readvariableop_resource:4
&conv_3_biasadd_readvariableop_resource:1
#batchnorm_3_readvariableop_resource:3
%batchnorm_3_readvariableop_1_resource:B
4batchnorm_3_fusedbatchnormv3_readvariableop_resource:D
6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource:@
&conv_4a_conv2d_readvariableop_resource:5
'conv_4a_biasadd_readvariableop_resource:2
$batchnorm_4a_readvariableop_resource:4
&batchnorm_4a_readvariableop_1_resource:C
5batchnorm_4a_fusedbatchnormv3_readvariableop_resource:E
7batchnorm_4a_fusedbatchnormv3_readvariableop_1_resource:S
9conv_transpose_1_conv2d_transpose_readvariableop_resource:>
0conv_transpose_1_biasadd_readvariableop_resource:1
#batchnorm_4_readvariableop_resource:3
%batchnorm_4_readvariableop_1_resource:B
4batchnorm_4_fusedbatchnormv3_readvariableop_resource:D
6batchnorm_4_fusedbatchnormv3_readvariableop_1_resource:S
9conv_transpose_2_conv2d_transpose_readvariableop_resource:>
0conv_transpose_2_biasadd_readvariableop_resource:1
#batchnorm_5_readvariableop_resource:3
%batchnorm_5_readvariableop_1_resource:B
4batchnorm_5_fusedbatchnormv3_readvariableop_resource:D
6batchnorm_5_fusedbatchnormv3_readvariableop_1_resource:T
:conv_transpose_30_conv2d_transpose_readvariableop_resource:?
1conv_transpose_30_biasadd_readvariableop_resource:2
$batchnorm_31_readvariableop_resource:4
&batchnorm_31_readvariableop_1_resource:C
5batchnorm_31_fusedbatchnormv3_readvariableop_resource:E
7batchnorm_31_fusedbatchnormv3_readvariableop_1_resource:S
9conv_transpose_a_conv2d_transpose_readvariableop_resource:>
0conv_transpose_a_biasadd_readvariableop_resource:1
#batchnorm_b_readvariableop_resource:3
%batchnorm_b_readvariableop_1_resource:B
4batchnorm_b_fusedbatchnormv3_readvariableop_resource:D
6batchnorm_b_fusedbatchnormv3_readvariableop_1_resource:T
:conv_transpose_4a_conv2d_transpose_readvariableop_resource:?
1conv_transpose_4a_biasadd_readvariableop_resource:
identity??batchnorm_1/AssignNewValue?batchnorm_1/AssignNewValue_1?+batchnorm_1/FusedBatchNormV3/ReadVariableOp?-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?batchnorm_1/ReadVariableOp?batchnorm_1/ReadVariableOp_1?batchnorm_2/AssignNewValue?batchnorm_2/AssignNewValue_1?+batchnorm_2/FusedBatchNormV3/ReadVariableOp?-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1?batchnorm_2/ReadVariableOp?batchnorm_2/ReadVariableOp_1?batchnorm_3/AssignNewValue?batchnorm_3/AssignNewValue_1?+batchnorm_3/FusedBatchNormV3/ReadVariableOp?-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1?batchnorm_3/ReadVariableOp?batchnorm_3/ReadVariableOp_1?batchnorm_31/AssignNewValue?batchnorm_31/AssignNewValue_1?,batchnorm_31/FusedBatchNormV3/ReadVariableOp?.batchnorm_31/FusedBatchNormV3/ReadVariableOp_1?batchnorm_31/ReadVariableOp?batchnorm_31/ReadVariableOp_1?batchnorm_4/AssignNewValue?batchnorm_4/AssignNewValue_1?+batchnorm_4/FusedBatchNormV3/ReadVariableOp?-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1?batchnorm_4/ReadVariableOp?batchnorm_4/ReadVariableOp_1?batchnorm_4a/AssignNewValue?batchnorm_4a/AssignNewValue_1?,batchnorm_4a/FusedBatchNormV3/ReadVariableOp?.batchnorm_4a/FusedBatchNormV3/ReadVariableOp_1?batchnorm_4a/ReadVariableOp?batchnorm_4a/ReadVariableOp_1?batchnorm_5/AssignNewValue?batchnorm_5/AssignNewValue_1?+batchnorm_5/FusedBatchNormV3/ReadVariableOp?-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1?batchnorm_5/ReadVariableOp?batchnorm_5/ReadVariableOp_1?batchnorm_b/AssignNewValue?batchnorm_b/AssignNewValue_1?+batchnorm_b/FusedBatchNormV3/ReadVariableOp?-batchnorm_b/FusedBatchNormV3/ReadVariableOp_1?batchnorm_b/ReadVariableOp?batchnorm_b/ReadVariableOp_1?conv_1/BiasAdd/ReadVariableOp?conv_1/Conv2D/ReadVariableOp?conv_2/BiasAdd/ReadVariableOp?conv_2/Conv2D/ReadVariableOp?conv_3/BiasAdd/ReadVariableOp?conv_3/Conv2D/ReadVariableOp?conv_4a/BiasAdd/ReadVariableOp?conv_4a/Conv2D/ReadVariableOp?'conv_transpose_1/BiasAdd/ReadVariableOp?0conv_transpose_1/conv2d_transpose/ReadVariableOp?'conv_transpose_2/BiasAdd/ReadVariableOp?0conv_transpose_2/conv2d_transpose/ReadVariableOp?(conv_transpose_30/BiasAdd/ReadVariableOp?1conv_transpose_30/conv2d_transpose/ReadVariableOp?(conv_transpose_4a/BiasAdd/ReadVariableOp?1conv_transpose_4a/conv2d_transpose/ReadVariableOp?'conv_transpose_a/BiasAdd/ReadVariableOp?0conv_transpose_a/conv2d_transpose/ReadVariableOp?
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv_1/Conv2DConv2Dinputs$conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????z
batchnorm_1/ReadVariableOpReadVariableOp#batchnorm_1_readvariableop_resource*
_output_shapes
:*
dtype0~
batchnorm_1/ReadVariableOp_1ReadVariableOp%batchnorm_1_readvariableop_1_resource*
_output_shapes
:*
dtype0?
+batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
batchnorm_1/FusedBatchNormV3FusedBatchNormV3conv_1/BiasAdd:output:0"batchnorm_1/ReadVariableOp:value:0$batchnorm_1/ReadVariableOp_1:value:03batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
batchnorm_1/AssignNewValueAssignVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource)batchnorm_1/FusedBatchNormV3:batch_mean:0,^batchnorm_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
batchnorm_1/AssignNewValue_1AssignVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource-batchnorm_1/FusedBatchNormV3:batch_variance:0.^batchnorm_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_relu_1/LeakyRelu	LeakyRelu batchnorm_1/FusedBatchNormV3:y:0*1
_output_shapes
:???????????*
alpha%???>?
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv_2/Conv2DConv2D$leaky_relu_1/LeakyRelu:activations:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
?
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  z
batchnorm_2/ReadVariableOpReadVariableOp#batchnorm_2_readvariableop_resource*
_output_shapes
:*
dtype0~
batchnorm_2/ReadVariableOp_1ReadVariableOp%batchnorm_2_readvariableop_1_resource*
_output_shapes
:*
dtype0?
+batchnorm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
batchnorm_2/FusedBatchNormV3FusedBatchNormV3conv_2/BiasAdd:output:0"batchnorm_2/ReadVariableOp:value:0$batchnorm_2/ReadVariableOp_1:value:03batchnorm_2/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
batchnorm_2/AssignNewValueAssignVariableOp4batchnorm_2_fusedbatchnormv3_readvariableop_resource)batchnorm_2/FusedBatchNormV3:batch_mean:0,^batchnorm_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
batchnorm_2/AssignNewValue_1AssignVariableOp6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource-batchnorm_2/FusedBatchNormV3:batch_variance:0.^batchnorm_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_relu_2/LeakyRelu	LeakyRelu batchnorm_2/FusedBatchNormV3:y:0*/
_output_shapes
:?????????  *
alpha%???>?
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv_3/Conv2DConv2D$leaky_relu_2/LeakyRelu:activations:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z
batchnorm_3/ReadVariableOpReadVariableOp#batchnorm_3_readvariableop_resource*
_output_shapes
:*
dtype0~
batchnorm_3/ReadVariableOp_1ReadVariableOp%batchnorm_3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
+batchnorm_3/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
batchnorm_3/FusedBatchNormV3FusedBatchNormV3conv_3/BiasAdd:output:0"batchnorm_3/ReadVariableOp:value:0$batchnorm_3/ReadVariableOp_1:value:03batchnorm_3/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
batchnorm_3/AssignNewValueAssignVariableOp4batchnorm_3_fusedbatchnormv3_readvariableop_resource)batchnorm_3/FusedBatchNormV3:batch_mean:0,^batchnorm_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
batchnorm_3/AssignNewValue_1AssignVariableOp6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource-batchnorm_3/FusedBatchNormV3:batch_variance:0.^batchnorm_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_relu_3/LeakyRelu	LeakyRelu batchnorm_3/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>?
conv_4a/Conv2D/ReadVariableOpReadVariableOp&conv_4a_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv_4a/Conv2DConv2D$leaky_relu_3/LeakyRelu:activations:0%conv_4a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv_4a/BiasAdd/ReadVariableOpReadVariableOp'conv_4a_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_4a/BiasAddBiasAddconv_4a/Conv2D:output:0&conv_4a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????|
batchnorm_4a/ReadVariableOpReadVariableOp$batchnorm_4a_readvariableop_resource*
_output_shapes
:*
dtype0?
batchnorm_4a/ReadVariableOp_1ReadVariableOp&batchnorm_4a_readvariableop_1_resource*
_output_shapes
:*
dtype0?
,batchnorm_4a/FusedBatchNormV3/ReadVariableOpReadVariableOp5batchnorm_4a_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
.batchnorm_4a/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7batchnorm_4a_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
batchnorm_4a/FusedBatchNormV3FusedBatchNormV3conv_4a/BiasAdd:output:0#batchnorm_4a/ReadVariableOp:value:0%batchnorm_4a/ReadVariableOp_1:value:04batchnorm_4a/FusedBatchNormV3/ReadVariableOp:value:06batchnorm_4a/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
batchnorm_4a/AssignNewValueAssignVariableOp5batchnorm_4a_fusedbatchnormv3_readvariableop_resource*batchnorm_4a/FusedBatchNormV3:batch_mean:0-^batchnorm_4a/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
batchnorm_4a/AssignNewValue_1AssignVariableOp7batchnorm_4a_fusedbatchnormv3_readvariableop_1_resource.batchnorm_4a/FusedBatchNormV3:batch_variance:0/^batchnorm_4a/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_relu_4a/LeakyRelu	LeakyRelu!batchnorm_4a/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>k
conv_transpose_1/ShapeShape%leaky_relu_4a/LeakyRelu:activations:0*
T0*
_output_shapes
:n
$conv_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv_transpose_1/strided_sliceStridedSliceconv_transpose_1/Shape:output:0-conv_transpose_1/strided_slice/stack:output:0/conv_transpose_1/strided_slice/stack_1:output:0/conv_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv_transpose_1/stackPack'conv_transpose_1/strided_slice:output:0!conv_transpose_1/stack/1:output:0!conv_transpose_1/stack/2:output:0!conv_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv_transpose_1/strided_slice_1StridedSliceconv_transpose_1/stack:output:0/conv_transpose_1/strided_slice_1/stack:output:01conv_transpose_1/strided_slice_1/stack_1:output:01conv_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
!conv_transpose_1/conv2d_transposeConv2DBackpropInputconv_transpose_1/stack:output:08conv_transpose_1/conv2d_transpose/ReadVariableOp:value:0%leaky_relu_4a/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
'conv_transpose_1/BiasAdd/ReadVariableOpReadVariableOp0conv_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_transpose_1/BiasAddBiasAdd*conv_transpose_1/conv2d_transpose:output:0/conv_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z
batchnorm_4/ReadVariableOpReadVariableOp#batchnorm_4_readvariableop_resource*
_output_shapes
:*
dtype0~
batchnorm_4/ReadVariableOp_1ReadVariableOp%batchnorm_4_readvariableop_1_resource*
_output_shapes
:*
dtype0?
+batchnorm_4/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
batchnorm_4/FusedBatchNormV3FusedBatchNormV3!conv_transpose_1/BiasAdd:output:0"batchnorm_4/ReadVariableOp:value:0$batchnorm_4/ReadVariableOp_1:value:03batchnorm_4/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
batchnorm_4/AssignNewValueAssignVariableOp4batchnorm_4_fusedbatchnormv3_readvariableop_resource)batchnorm_4/FusedBatchNormV3:batch_mean:0,^batchnorm_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
batchnorm_4/AssignNewValue_1AssignVariableOp6batchnorm_4_fusedbatchnormv3_readvariableop_1_resource-batchnorm_4/FusedBatchNormV3:batch_variance:0.^batchnorm_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_relu_4/LeakyRelu	LeakyRelu batchnorm_4/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>j
conv_transpose_2/ShapeShape$leaky_relu_4/LeakyRelu:activations:0*
T0*
_output_shapes
:n
$conv_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv_transpose_2/strided_sliceStridedSliceconv_transpose_2/Shape:output:0-conv_transpose_2/strided_slice/stack:output:0/conv_transpose_2/strided_slice/stack_1:output:0/conv_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv_transpose_2/stackPack'conv_transpose_2/strided_slice:output:0!conv_transpose_2/stack/1:output:0!conv_transpose_2/stack/2:output:0!conv_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv_transpose_2/strided_slice_1StridedSliceconv_transpose_2/stack:output:0/conv_transpose_2/strided_slice_1/stack:output:01conv_transpose_2/strided_slice_1/stack_1:output:01conv_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
!conv_transpose_2/conv2d_transposeConv2DBackpropInputconv_transpose_2/stack:output:08conv_transpose_2/conv2d_transpose/ReadVariableOp:value:0$leaky_relu_4/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
'conv_transpose_2/BiasAdd/ReadVariableOpReadVariableOp0conv_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_transpose_2/BiasAddBiasAdd*conv_transpose_2/conv2d_transpose:output:0/conv_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z
batchnorm_5/ReadVariableOpReadVariableOp#batchnorm_5_readvariableop_resource*
_output_shapes
:*
dtype0~
batchnorm_5/ReadVariableOp_1ReadVariableOp%batchnorm_5_readvariableop_1_resource*
_output_shapes
:*
dtype0?
+batchnorm_5/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
batchnorm_5/FusedBatchNormV3FusedBatchNormV3!conv_transpose_2/BiasAdd:output:0"batchnorm_5/ReadVariableOp:value:0$batchnorm_5/ReadVariableOp_1:value:03batchnorm_5/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
batchnorm_5/AssignNewValueAssignVariableOp4batchnorm_5_fusedbatchnormv3_readvariableop_resource)batchnorm_5/FusedBatchNormV3:batch_mean:0,^batchnorm_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
batchnorm_5/AssignNewValue_1AssignVariableOp6batchnorm_5_fusedbatchnormv3_readvariableop_1_resource-batchnorm_5/FusedBatchNormV3:batch_variance:0.^batchnorm_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_relu_5/LeakyRelu	LeakyRelu batchnorm_5/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>k
conv_transpose_30/ShapeShape$leaky_relu_5/LeakyRelu:activations:0*
T0*
_output_shapes
:o
%conv_transpose_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'conv_transpose_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'conv_transpose_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv_transpose_30/strided_sliceStridedSlice conv_transpose_30/Shape:output:0.conv_transpose_30/strided_slice/stack:output:00conv_transpose_30/strided_slice/stack_1:output:00conv_transpose_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
conv_transpose_30/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@[
conv_transpose_30/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@[
conv_transpose_30/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv_transpose_30/stackPack(conv_transpose_30/strided_slice:output:0"conv_transpose_30/stack/1:output:0"conv_transpose_30/stack/2:output:0"conv_transpose_30/stack/3:output:0*
N*
T0*
_output_shapes
:q
'conv_transpose_30/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv_transpose_30/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv_transpose_30/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv_transpose_30/strided_slice_1StridedSlice conv_transpose_30/stack:output:00conv_transpose_30/strided_slice_1/stack:output:02conv_transpose_30/strided_slice_1/stack_1:output:02conv_transpose_30/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1conv_transpose_30/conv2d_transpose/ReadVariableOpReadVariableOp:conv_transpose_30_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
"conv_transpose_30/conv2d_transposeConv2DBackpropInput conv_transpose_30/stack:output:09conv_transpose_30/conv2d_transpose/ReadVariableOp:value:0$leaky_relu_5/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
(conv_transpose_30/BiasAdd/ReadVariableOpReadVariableOp1conv_transpose_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_transpose_30/BiasAddBiasAdd+conv_transpose_30/conv2d_transpose:output:00conv_transpose_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@|
batchnorm_31/ReadVariableOpReadVariableOp$batchnorm_31_readvariableop_resource*
_output_shapes
:*
dtype0?
batchnorm_31/ReadVariableOp_1ReadVariableOp&batchnorm_31_readvariableop_1_resource*
_output_shapes
:*
dtype0?
,batchnorm_31/FusedBatchNormV3/ReadVariableOpReadVariableOp5batchnorm_31_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
.batchnorm_31/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7batchnorm_31_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
batchnorm_31/FusedBatchNormV3FusedBatchNormV3"conv_transpose_30/BiasAdd:output:0#batchnorm_31/ReadVariableOp:value:0%batchnorm_31/ReadVariableOp_1:value:04batchnorm_31/FusedBatchNormV3/ReadVariableOp:value:06batchnorm_31/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
batchnorm_31/AssignNewValueAssignVariableOp5batchnorm_31_fusedbatchnormv3_readvariableop_resource*batchnorm_31/FusedBatchNormV3:batch_mean:0-^batchnorm_31/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
batchnorm_31/AssignNewValue_1AssignVariableOp7batchnorm_31_fusedbatchnormv3_readvariableop_1_resource.batchnorm_31/FusedBatchNormV3:batch_variance:0/^batchnorm_31/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_relu_32/LeakyRelu	LeakyRelu!batchnorm_31/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@*
alpha%???>k
conv_transpose_a/ShapeShape%leaky_relu_32/LeakyRelu:activations:0*
T0*
_output_shapes
:n
$conv_transpose_a/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv_transpose_a/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv_transpose_a/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv_transpose_a/strided_sliceStridedSliceconv_transpose_a/Shape:output:0-conv_transpose_a/strided_slice/stack:output:0/conv_transpose_a/strided_slice/stack_1:output:0/conv_transpose_a/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
conv_transpose_a/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?[
conv_transpose_a/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?Z
conv_transpose_a/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv_transpose_a/stackPack'conv_transpose_a/strided_slice:output:0!conv_transpose_a/stack/1:output:0!conv_transpose_a/stack/2:output:0!conv_transpose_a/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv_transpose_a/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv_transpose_a/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv_transpose_a/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv_transpose_a/strided_slice_1StridedSliceconv_transpose_a/stack:output:0/conv_transpose_a/strided_slice_1/stack:output:01conv_transpose_a/strided_slice_1/stack_1:output:01conv_transpose_a/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv_transpose_a/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_a_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
!conv_transpose_a/conv2d_transposeConv2DBackpropInputconv_transpose_a/stack:output:08conv_transpose_a/conv2d_transpose/ReadVariableOp:value:0%leaky_relu_32/LeakyRelu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
'conv_transpose_a/BiasAdd/ReadVariableOpReadVariableOp0conv_transpose_a_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_transpose_a/BiasAddBiasAdd*conv_transpose_a/conv2d_transpose:output:0/conv_transpose_a/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????z
batchnorm_b/ReadVariableOpReadVariableOp#batchnorm_b_readvariableop_resource*
_output_shapes
:*
dtype0~
batchnorm_b/ReadVariableOp_1ReadVariableOp%batchnorm_b_readvariableop_1_resource*
_output_shapes
:*
dtype0?
+batchnorm_b/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_b_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
-batchnorm_b/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_b_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
batchnorm_b/FusedBatchNormV3FusedBatchNormV3!conv_transpose_a/BiasAdd:output:0"batchnorm_b/ReadVariableOp:value:0$batchnorm_b/ReadVariableOp_1:value:03batchnorm_b/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_b/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
batchnorm_b/AssignNewValueAssignVariableOp4batchnorm_b_fusedbatchnormv3_readvariableop_resource)batchnorm_b/FusedBatchNormV3:batch_mean:0,^batchnorm_b/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
batchnorm_b/AssignNewValue_1AssignVariableOp6batchnorm_b_fusedbatchnormv3_readvariableop_1_resource-batchnorm_b/FusedBatchNormV3:batch_variance:0.^batchnorm_b/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_relu_c/LeakyRelu	LeakyRelu batchnorm_b/FusedBatchNormV3:y:0*1
_output_shapes
:???????????*
alpha%???>k
conv_transpose_4a/ShapeShape$leaky_relu_c/LeakyRelu:activations:0*
T0*
_output_shapes
:o
%conv_transpose_4a/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'conv_transpose_4a/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'conv_transpose_4a/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv_transpose_4a/strided_sliceStridedSlice conv_transpose_4a/Shape:output:0.conv_transpose_4a/strided_slice/stack:output:00conv_transpose_4a/strided_slice/stack_1:output:00conv_transpose_4a/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv_transpose_4a/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?\
conv_transpose_4a/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?[
conv_transpose_4a/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv_transpose_4a/stackPack(conv_transpose_4a/strided_slice:output:0"conv_transpose_4a/stack/1:output:0"conv_transpose_4a/stack/2:output:0"conv_transpose_4a/stack/3:output:0*
N*
T0*
_output_shapes
:q
'conv_transpose_4a/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv_transpose_4a/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv_transpose_4a/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv_transpose_4a/strided_slice_1StridedSlice conv_transpose_4a/stack:output:00conv_transpose_4a/strided_slice_1/stack:output:02conv_transpose_4a/strided_slice_1/stack_1:output:02conv_transpose_4a/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1conv_transpose_4a/conv2d_transpose/ReadVariableOpReadVariableOp:conv_transpose_4a_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
"conv_transpose_4a/conv2d_transposeConv2DBackpropInput conv_transpose_4a/stack:output:09conv_transpose_4a/conv2d_transpose/ReadVariableOp:value:0$leaky_relu_c/LeakyRelu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
(conv_transpose_4a/BiasAdd/ReadVariableOpReadVariableOp1conv_transpose_4a_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_transpose_4a/BiasAddBiasAdd+conv_transpose_4a/conv2d_transpose:output:00conv_transpose_4a/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
conv_transpose_4a/SigmoidSigmoid"conv_transpose_4a/BiasAdd:output:0*
T0*1
_output_shapes
:???????????v
IdentityIdentityconv_transpose_4a/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp^batchnorm_1/AssignNewValue^batchnorm_1/AssignNewValue_1,^batchnorm_1/FusedBatchNormV3/ReadVariableOp.^batchnorm_1/FusedBatchNormV3/ReadVariableOp_1^batchnorm_1/ReadVariableOp^batchnorm_1/ReadVariableOp_1^batchnorm_2/AssignNewValue^batchnorm_2/AssignNewValue_1,^batchnorm_2/FusedBatchNormV3/ReadVariableOp.^batchnorm_2/FusedBatchNormV3/ReadVariableOp_1^batchnorm_2/ReadVariableOp^batchnorm_2/ReadVariableOp_1^batchnorm_3/AssignNewValue^batchnorm_3/AssignNewValue_1,^batchnorm_3/FusedBatchNormV3/ReadVariableOp.^batchnorm_3/FusedBatchNormV3/ReadVariableOp_1^batchnorm_3/ReadVariableOp^batchnorm_3/ReadVariableOp_1^batchnorm_31/AssignNewValue^batchnorm_31/AssignNewValue_1-^batchnorm_31/FusedBatchNormV3/ReadVariableOp/^batchnorm_31/FusedBatchNormV3/ReadVariableOp_1^batchnorm_31/ReadVariableOp^batchnorm_31/ReadVariableOp_1^batchnorm_4/AssignNewValue^batchnorm_4/AssignNewValue_1,^batchnorm_4/FusedBatchNormV3/ReadVariableOp.^batchnorm_4/FusedBatchNormV3/ReadVariableOp_1^batchnorm_4/ReadVariableOp^batchnorm_4/ReadVariableOp_1^batchnorm_4a/AssignNewValue^batchnorm_4a/AssignNewValue_1-^batchnorm_4a/FusedBatchNormV3/ReadVariableOp/^batchnorm_4a/FusedBatchNormV3/ReadVariableOp_1^batchnorm_4a/ReadVariableOp^batchnorm_4a/ReadVariableOp_1^batchnorm_5/AssignNewValue^batchnorm_5/AssignNewValue_1,^batchnorm_5/FusedBatchNormV3/ReadVariableOp.^batchnorm_5/FusedBatchNormV3/ReadVariableOp_1^batchnorm_5/ReadVariableOp^batchnorm_5/ReadVariableOp_1^batchnorm_b/AssignNewValue^batchnorm_b/AssignNewValue_1,^batchnorm_b/FusedBatchNormV3/ReadVariableOp.^batchnorm_b/FusedBatchNormV3/ReadVariableOp_1^batchnorm_b/ReadVariableOp^batchnorm_b/ReadVariableOp_1^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^conv_4a/BiasAdd/ReadVariableOp^conv_4a/Conv2D/ReadVariableOp(^conv_transpose_1/BiasAdd/ReadVariableOp1^conv_transpose_1/conv2d_transpose/ReadVariableOp(^conv_transpose_2/BiasAdd/ReadVariableOp1^conv_transpose_2/conv2d_transpose/ReadVariableOp)^conv_transpose_30/BiasAdd/ReadVariableOp2^conv_transpose_30/conv2d_transpose/ReadVariableOp)^conv_transpose_4a/BiasAdd/ReadVariableOp2^conv_transpose_4a/conv2d_transpose/ReadVariableOp(^conv_transpose_a/BiasAdd/ReadVariableOp1^conv_transpose_a/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 28
batchnorm_1/AssignNewValuebatchnorm_1/AssignNewValue2<
batchnorm_1/AssignNewValue_1batchnorm_1/AssignNewValue_12Z
+batchnorm_1/FusedBatchNormV3/ReadVariableOp+batchnorm_1/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1-batchnorm_1/FusedBatchNormV3/ReadVariableOp_128
batchnorm_1/ReadVariableOpbatchnorm_1/ReadVariableOp2<
batchnorm_1/ReadVariableOp_1batchnorm_1/ReadVariableOp_128
batchnorm_2/AssignNewValuebatchnorm_2/AssignNewValue2<
batchnorm_2/AssignNewValue_1batchnorm_2/AssignNewValue_12Z
+batchnorm_2/FusedBatchNormV3/ReadVariableOp+batchnorm_2/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1-batchnorm_2/FusedBatchNormV3/ReadVariableOp_128
batchnorm_2/ReadVariableOpbatchnorm_2/ReadVariableOp2<
batchnorm_2/ReadVariableOp_1batchnorm_2/ReadVariableOp_128
batchnorm_3/AssignNewValuebatchnorm_3/AssignNewValue2<
batchnorm_3/AssignNewValue_1batchnorm_3/AssignNewValue_12Z
+batchnorm_3/FusedBatchNormV3/ReadVariableOp+batchnorm_3/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1-batchnorm_3/FusedBatchNormV3/ReadVariableOp_128
batchnorm_3/ReadVariableOpbatchnorm_3/ReadVariableOp2<
batchnorm_3/ReadVariableOp_1batchnorm_3/ReadVariableOp_12:
batchnorm_31/AssignNewValuebatchnorm_31/AssignNewValue2>
batchnorm_31/AssignNewValue_1batchnorm_31/AssignNewValue_12\
,batchnorm_31/FusedBatchNormV3/ReadVariableOp,batchnorm_31/FusedBatchNormV3/ReadVariableOp2`
.batchnorm_31/FusedBatchNormV3/ReadVariableOp_1.batchnorm_31/FusedBatchNormV3/ReadVariableOp_12:
batchnorm_31/ReadVariableOpbatchnorm_31/ReadVariableOp2>
batchnorm_31/ReadVariableOp_1batchnorm_31/ReadVariableOp_128
batchnorm_4/AssignNewValuebatchnorm_4/AssignNewValue2<
batchnorm_4/AssignNewValue_1batchnorm_4/AssignNewValue_12Z
+batchnorm_4/FusedBatchNormV3/ReadVariableOp+batchnorm_4/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1-batchnorm_4/FusedBatchNormV3/ReadVariableOp_128
batchnorm_4/ReadVariableOpbatchnorm_4/ReadVariableOp2<
batchnorm_4/ReadVariableOp_1batchnorm_4/ReadVariableOp_12:
batchnorm_4a/AssignNewValuebatchnorm_4a/AssignNewValue2>
batchnorm_4a/AssignNewValue_1batchnorm_4a/AssignNewValue_12\
,batchnorm_4a/FusedBatchNormV3/ReadVariableOp,batchnorm_4a/FusedBatchNormV3/ReadVariableOp2`
.batchnorm_4a/FusedBatchNormV3/ReadVariableOp_1.batchnorm_4a/FusedBatchNormV3/ReadVariableOp_12:
batchnorm_4a/ReadVariableOpbatchnorm_4a/ReadVariableOp2>
batchnorm_4a/ReadVariableOp_1batchnorm_4a/ReadVariableOp_128
batchnorm_5/AssignNewValuebatchnorm_5/AssignNewValue2<
batchnorm_5/AssignNewValue_1batchnorm_5/AssignNewValue_12Z
+batchnorm_5/FusedBatchNormV3/ReadVariableOp+batchnorm_5/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1-batchnorm_5/FusedBatchNormV3/ReadVariableOp_128
batchnorm_5/ReadVariableOpbatchnorm_5/ReadVariableOp2<
batchnorm_5/ReadVariableOp_1batchnorm_5/ReadVariableOp_128
batchnorm_b/AssignNewValuebatchnorm_b/AssignNewValue2<
batchnorm_b/AssignNewValue_1batchnorm_b/AssignNewValue_12Z
+batchnorm_b/FusedBatchNormV3/ReadVariableOp+batchnorm_b/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_b/FusedBatchNormV3/ReadVariableOp_1-batchnorm_b/FusedBatchNormV3/ReadVariableOp_128
batchnorm_b/ReadVariableOpbatchnorm_b/ReadVariableOp2<
batchnorm_b/ReadVariableOp_1batchnorm_b/ReadVariableOp_12>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp2@
conv_4a/BiasAdd/ReadVariableOpconv_4a/BiasAdd/ReadVariableOp2>
conv_4a/Conv2D/ReadVariableOpconv_4a/Conv2D/ReadVariableOp2R
'conv_transpose_1/BiasAdd/ReadVariableOp'conv_transpose_1/BiasAdd/ReadVariableOp2d
0conv_transpose_1/conv2d_transpose/ReadVariableOp0conv_transpose_1/conv2d_transpose/ReadVariableOp2R
'conv_transpose_2/BiasAdd/ReadVariableOp'conv_transpose_2/BiasAdd/ReadVariableOp2d
0conv_transpose_2/conv2d_transpose/ReadVariableOp0conv_transpose_2/conv2d_transpose/ReadVariableOp2T
(conv_transpose_30/BiasAdd/ReadVariableOp(conv_transpose_30/BiasAdd/ReadVariableOp2f
1conv_transpose_30/conv2d_transpose/ReadVariableOp1conv_transpose_30/conv2d_transpose/ReadVariableOp2T
(conv_transpose_4a/BiasAdd/ReadVariableOp(conv_transpose_4a/BiasAdd/ReadVariableOp2f
1conv_transpose_4a/conv2d_transpose/ReadVariableOp1conv_transpose_4a/conv2d_transpose/ReadVariableOp2R
'conv_transpose_a/BiasAdd/ReadVariableOp'conv_transpose_a/BiasAdd/ReadVariableOp2d
0conv_transpose_a/conv2d_transpose/ReadVariableOp0conv_transpose_a/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
A__inference_conv_4a_layer_call_and_return_conditional_losses_2338

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
?
J__inference_conv_transpose_1_layer_call_and_return_conditional_losses_4554

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_1_layer_call_fn_4180

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_1514?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_4148
input_layer!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:$

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:$

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:$

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:$

unknown_47:

unknown_48:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_1492y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameinput_layer
?
?
+__inference_batchnorm_31_layer_call_fn_4808

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_batchnorm_31_layer_call_and_return_conditional_losses_2061?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_leaky_relu_1_layer_call_and_return_conditional_losses_4239

inputs
identitya
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:???????????*
alpha%???>i
IdentityIdentityLeakyRelu:activations:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_4a_layer_call_and_return_conditional_losses_4484

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_3_layer_call_fn_4362

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_1642?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
@__inference_conv_3_layer_call_and_return_conditional_losses_4349

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_3448

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:$

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:$

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:$

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:$

unknown_47:

unknown_48:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_2450y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_4712

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_4229

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
@__inference_conv_3_layer_call_and_return_conditional_losses_2306

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_4320

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_leaky_relu_c_layer_call_and_return_conditional_losses_2442

inputs
identitya
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:???????????*
alpha%???>i
IdentityIdentityLeakyRelu:activations:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
? 
?
J__inference_conv_transpose_1_layer_call_and_return_conditional_losses_1785

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_1922

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
@__inference_conv_2_layer_call_and_return_conditional_losses_2274

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????  w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_leaky_relu_4_layer_call_and_return_conditional_losses_4626

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_5_layer_call_fn_4694

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_1953?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_leaky_relu_5_layer_call_fn_4735

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_5_layer_call_and_return_conditional_losses_2400h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
@__inference_conv_1_layer_call_and_return_conditional_losses_2242

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_1609

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
0__inference_conv_transpose_30_layer_call_fn_4749

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv_transpose_30_layer_call_and_return_conditional_losses_2001?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_leaky_relu_32_layer_call_and_return_conditional_losses_2421

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@@*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
? 
?
J__inference_conv_transpose_a_layer_call_and_return_conditional_losses_2109

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_5_layer_call_fn_4681

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_1922?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_3_layer_call_fn_4375

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_1673?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_1_layer_call_fn_4193

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_1545?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_4a_layer_call_and_return_conditional_losses_4502

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_batchnorm_31_layer_call_and_return_conditional_losses_4844

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_4a_layer_call_fn_4453

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_batchnorm_4a_layer_call_and_return_conditional_losses_1706?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_1578

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_4_layer_call_fn_4567

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_1814?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_leaky_relu_32_layer_call_and_return_conditional_losses_4854

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@@*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?

?
@__inference_conv_1_layer_call_and_return_conditional_losses_4167

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
H
,__inference_leaky_relu_4a_layer_call_fn_4507

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_leaky_relu_4a_layer_call_and_return_conditional_losses_2358h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_conv_transpose_4a_layer_call_fn_4977

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv_transpose_4a_layer_call_and_return_conditional_losses_2218?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_batchnorm_31_layer_call_fn_4795

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_batchnorm_31_layer_call_and_return_conditional_losses_2030?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_b_layer_call_and_return_conditional_losses_4940

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
/__inference_conv_transpose_1_layer_call_fn_4521

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_1_layer_call_and_return_conditional_losses_1785?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
%__inference_conv_1_layer_call_fn_4157

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_1_layer_call_and_return_conditional_losses_2242y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?!
?
K__inference_conv_transpose_4a_layer_call_and_return_conditional_losses_2218

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_4598

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_3085
input_layer!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:$

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:$

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:$

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:$

unknown_47:

unknown_48:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*D
_read_only_resource_inputs&
$"	
 !"%&'(+,-.12*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_2877y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameinput_layer
?
?
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_4616

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?-
__inference__wrapped_model_1492
input_layerE
+model_conv_1_conv2d_readvariableop_resource::
,model_conv_1_biasadd_readvariableop_resource:7
)model_batchnorm_1_readvariableop_resource:9
+model_batchnorm_1_readvariableop_1_resource:H
:model_batchnorm_1_fusedbatchnormv3_readvariableop_resource:J
<model_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource:E
+model_conv_2_conv2d_readvariableop_resource::
,model_conv_2_biasadd_readvariableop_resource:7
)model_batchnorm_2_readvariableop_resource:9
+model_batchnorm_2_readvariableop_1_resource:H
:model_batchnorm_2_fusedbatchnormv3_readvariableop_resource:J
<model_batchnorm_2_fusedbatchnormv3_readvariableop_1_resource:E
+model_conv_3_conv2d_readvariableop_resource::
,model_conv_3_biasadd_readvariableop_resource:7
)model_batchnorm_3_readvariableop_resource:9
+model_batchnorm_3_readvariableop_1_resource:H
:model_batchnorm_3_fusedbatchnormv3_readvariableop_resource:J
<model_batchnorm_3_fusedbatchnormv3_readvariableop_1_resource:F
,model_conv_4a_conv2d_readvariableop_resource:;
-model_conv_4a_biasadd_readvariableop_resource:8
*model_batchnorm_4a_readvariableop_resource::
,model_batchnorm_4a_readvariableop_1_resource:I
;model_batchnorm_4a_fusedbatchnormv3_readvariableop_resource:K
=model_batchnorm_4a_fusedbatchnormv3_readvariableop_1_resource:Y
?model_conv_transpose_1_conv2d_transpose_readvariableop_resource:D
6model_conv_transpose_1_biasadd_readvariableop_resource:7
)model_batchnorm_4_readvariableop_resource:9
+model_batchnorm_4_readvariableop_1_resource:H
:model_batchnorm_4_fusedbatchnormv3_readvariableop_resource:J
<model_batchnorm_4_fusedbatchnormv3_readvariableop_1_resource:Y
?model_conv_transpose_2_conv2d_transpose_readvariableop_resource:D
6model_conv_transpose_2_biasadd_readvariableop_resource:7
)model_batchnorm_5_readvariableop_resource:9
+model_batchnorm_5_readvariableop_1_resource:H
:model_batchnorm_5_fusedbatchnormv3_readvariableop_resource:J
<model_batchnorm_5_fusedbatchnormv3_readvariableop_1_resource:Z
@model_conv_transpose_30_conv2d_transpose_readvariableop_resource:E
7model_conv_transpose_30_biasadd_readvariableop_resource:8
*model_batchnorm_31_readvariableop_resource::
,model_batchnorm_31_readvariableop_1_resource:I
;model_batchnorm_31_fusedbatchnormv3_readvariableop_resource:K
=model_batchnorm_31_fusedbatchnormv3_readvariableop_1_resource:Y
?model_conv_transpose_a_conv2d_transpose_readvariableop_resource:D
6model_conv_transpose_a_biasadd_readvariableop_resource:7
)model_batchnorm_b_readvariableop_resource:9
+model_batchnorm_b_readvariableop_1_resource:H
:model_batchnorm_b_fusedbatchnormv3_readvariableop_resource:J
<model_batchnorm_b_fusedbatchnormv3_readvariableop_1_resource:Z
@model_conv_transpose_4a_conv2d_transpose_readvariableop_resource:E
7model_conv_transpose_4a_biasadd_readvariableop_resource:
identity??1model/batchnorm_1/FusedBatchNormV3/ReadVariableOp?3model/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1? model/batchnorm_1/ReadVariableOp?"model/batchnorm_1/ReadVariableOp_1?1model/batchnorm_2/FusedBatchNormV3/ReadVariableOp?3model/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1? model/batchnorm_2/ReadVariableOp?"model/batchnorm_2/ReadVariableOp_1?1model/batchnorm_3/FusedBatchNormV3/ReadVariableOp?3model/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1? model/batchnorm_3/ReadVariableOp?"model/batchnorm_3/ReadVariableOp_1?2model/batchnorm_31/FusedBatchNormV3/ReadVariableOp?4model/batchnorm_31/FusedBatchNormV3/ReadVariableOp_1?!model/batchnorm_31/ReadVariableOp?#model/batchnorm_31/ReadVariableOp_1?1model/batchnorm_4/FusedBatchNormV3/ReadVariableOp?3model/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1? model/batchnorm_4/ReadVariableOp?"model/batchnorm_4/ReadVariableOp_1?2model/batchnorm_4a/FusedBatchNormV3/ReadVariableOp?4model/batchnorm_4a/FusedBatchNormV3/ReadVariableOp_1?!model/batchnorm_4a/ReadVariableOp?#model/batchnorm_4a/ReadVariableOp_1?1model/batchnorm_5/FusedBatchNormV3/ReadVariableOp?3model/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1? model/batchnorm_5/ReadVariableOp?"model/batchnorm_5/ReadVariableOp_1?1model/batchnorm_b/FusedBatchNormV3/ReadVariableOp?3model/batchnorm_b/FusedBatchNormV3/ReadVariableOp_1? model/batchnorm_b/ReadVariableOp?"model/batchnorm_b/ReadVariableOp_1?#model/conv_1/BiasAdd/ReadVariableOp?"model/conv_1/Conv2D/ReadVariableOp?#model/conv_2/BiasAdd/ReadVariableOp?"model/conv_2/Conv2D/ReadVariableOp?#model/conv_3/BiasAdd/ReadVariableOp?"model/conv_3/Conv2D/ReadVariableOp?$model/conv_4a/BiasAdd/ReadVariableOp?#model/conv_4a/Conv2D/ReadVariableOp?-model/conv_transpose_1/BiasAdd/ReadVariableOp?6model/conv_transpose_1/conv2d_transpose/ReadVariableOp?-model/conv_transpose_2/BiasAdd/ReadVariableOp?6model/conv_transpose_2/conv2d_transpose/ReadVariableOp?.model/conv_transpose_30/BiasAdd/ReadVariableOp?7model/conv_transpose_30/conv2d_transpose/ReadVariableOp?.model/conv_transpose_4a/BiasAdd/ReadVariableOp?7model/conv_transpose_4a/conv2d_transpose/ReadVariableOp?-model/conv_transpose_a/BiasAdd/ReadVariableOp?6model/conv_transpose_a/conv2d_transpose/ReadVariableOp?
"model/conv_1/Conv2D/ReadVariableOpReadVariableOp+model_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model/conv_1/Conv2DConv2Dinput_layer*model/conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
#model/conv_1/BiasAdd/ReadVariableOpReadVariableOp,model_conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/conv_1/BiasAddBiasAddmodel/conv_1/Conv2D:output:0+model/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
 model/batchnorm_1/ReadVariableOpReadVariableOp)model_batchnorm_1_readvariableop_resource*
_output_shapes
:*
dtype0?
"model/batchnorm_1/ReadVariableOp_1ReadVariableOp+model_batchnorm_1_readvariableop_1_resource*
_output_shapes
:*
dtype0?
1model/batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp:model_batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
3model/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<model_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
"model/batchnorm_1/FusedBatchNormV3FusedBatchNormV3model/conv_1/BiasAdd:output:0(model/batchnorm_1/ReadVariableOp:value:0*model/batchnorm_1/ReadVariableOp_1:value:09model/batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:0;model/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
model/leaky_relu_1/LeakyRelu	LeakyRelu&model/batchnorm_1/FusedBatchNormV3:y:0*1
_output_shapes
:???????????*
alpha%???>?
"model/conv_2/Conv2D/ReadVariableOpReadVariableOp+model_conv_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model/conv_2/Conv2DConv2D*model/leaky_relu_1/LeakyRelu:activations:0*model/conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
?
#model/conv_2/BiasAdd/ReadVariableOpReadVariableOp,model_conv_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/conv_2/BiasAddBiasAddmodel/conv_2/Conv2D:output:0+model/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  ?
 model/batchnorm_2/ReadVariableOpReadVariableOp)model_batchnorm_2_readvariableop_resource*
_output_shapes
:*
dtype0?
"model/batchnorm_2/ReadVariableOp_1ReadVariableOp+model_batchnorm_2_readvariableop_1_resource*
_output_shapes
:*
dtype0?
1model/batchnorm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp:model_batchnorm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
3model/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<model_batchnorm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
"model/batchnorm_2/FusedBatchNormV3FusedBatchNormV3model/conv_2/BiasAdd:output:0(model/batchnorm_2/ReadVariableOp:value:0*model/batchnorm_2/ReadVariableOp_1:value:09model/batchnorm_2/FusedBatchNormV3/ReadVariableOp:value:0;model/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( ?
model/leaky_relu_2/LeakyRelu	LeakyRelu&model/batchnorm_2/FusedBatchNormV3:y:0*/
_output_shapes
:?????????  *
alpha%???>?
"model/conv_3/Conv2D/ReadVariableOpReadVariableOp+model_conv_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model/conv_3/Conv2DConv2D*model/leaky_relu_2/LeakyRelu:activations:0*model/conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
#model/conv_3/BiasAdd/ReadVariableOpReadVariableOp,model_conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/conv_3/BiasAddBiasAddmodel/conv_3/Conv2D:output:0+model/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
 model/batchnorm_3/ReadVariableOpReadVariableOp)model_batchnorm_3_readvariableop_resource*
_output_shapes
:*
dtype0?
"model/batchnorm_3/ReadVariableOp_1ReadVariableOp+model_batchnorm_3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
1model/batchnorm_3/FusedBatchNormV3/ReadVariableOpReadVariableOp:model_batchnorm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
3model/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<model_batchnorm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
"model/batchnorm_3/FusedBatchNormV3FusedBatchNormV3model/conv_3/BiasAdd:output:0(model/batchnorm_3/ReadVariableOp:value:0*model/batchnorm_3/ReadVariableOp_1:value:09model/batchnorm_3/FusedBatchNormV3/ReadVariableOp:value:0;model/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
model/leaky_relu_3/LeakyRelu	LeakyRelu&model/batchnorm_3/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>?
#model/conv_4a/Conv2D/ReadVariableOpReadVariableOp,model_conv_4a_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model/conv_4a/Conv2DConv2D*model/leaky_relu_3/LeakyRelu:activations:0+model/conv_4a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
$model/conv_4a/BiasAdd/ReadVariableOpReadVariableOp-model_conv_4a_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/conv_4a/BiasAddBiasAddmodel/conv_4a/Conv2D:output:0,model/conv_4a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
!model/batchnorm_4a/ReadVariableOpReadVariableOp*model_batchnorm_4a_readvariableop_resource*
_output_shapes
:*
dtype0?
#model/batchnorm_4a/ReadVariableOp_1ReadVariableOp,model_batchnorm_4a_readvariableop_1_resource*
_output_shapes
:*
dtype0?
2model/batchnorm_4a/FusedBatchNormV3/ReadVariableOpReadVariableOp;model_batchnorm_4a_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
4model/batchnorm_4a/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=model_batchnorm_4a_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
#model/batchnorm_4a/FusedBatchNormV3FusedBatchNormV3model/conv_4a/BiasAdd:output:0)model/batchnorm_4a/ReadVariableOp:value:0+model/batchnorm_4a/ReadVariableOp_1:value:0:model/batchnorm_4a/FusedBatchNormV3/ReadVariableOp:value:0<model/batchnorm_4a/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
model/leaky_relu_4a/LeakyRelu	LeakyRelu'model/batchnorm_4a/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>w
model/conv_transpose_1/ShapeShape+model/leaky_relu_4a/LeakyRelu:activations:0*
T0*
_output_shapes
:t
*model/conv_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,model/conv_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,model/conv_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$model/conv_transpose_1/strided_sliceStridedSlice%model/conv_transpose_1/Shape:output:03model/conv_transpose_1/strided_slice/stack:output:05model/conv_transpose_1/strided_slice/stack_1:output:05model/conv_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
model/conv_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`
model/conv_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :`
model/conv_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
model/conv_transpose_1/stackPack-model/conv_transpose_1/strided_slice:output:0'model/conv_transpose_1/stack/1:output:0'model/conv_transpose_1/stack/2:output:0'model/conv_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:v
,model/conv_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/conv_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/conv_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/conv_transpose_1/strided_slice_1StridedSlice%model/conv_transpose_1/stack:output:05model/conv_transpose_1/strided_slice_1/stack:output:07model/conv_transpose_1/strided_slice_1/stack_1:output:07model/conv_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
6model/conv_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp?model_conv_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
'model/conv_transpose_1/conv2d_transposeConv2DBackpropInput%model/conv_transpose_1/stack:output:0>model/conv_transpose_1/conv2d_transpose/ReadVariableOp:value:0+model/leaky_relu_4a/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
-model/conv_transpose_1/BiasAdd/ReadVariableOpReadVariableOp6model_conv_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/conv_transpose_1/BiasAddBiasAdd0model/conv_transpose_1/conv2d_transpose:output:05model/conv_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
 model/batchnorm_4/ReadVariableOpReadVariableOp)model_batchnorm_4_readvariableop_resource*
_output_shapes
:*
dtype0?
"model/batchnorm_4/ReadVariableOp_1ReadVariableOp+model_batchnorm_4_readvariableop_1_resource*
_output_shapes
:*
dtype0?
1model/batchnorm_4/FusedBatchNormV3/ReadVariableOpReadVariableOp:model_batchnorm_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
3model/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<model_batchnorm_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
"model/batchnorm_4/FusedBatchNormV3FusedBatchNormV3'model/conv_transpose_1/BiasAdd:output:0(model/batchnorm_4/ReadVariableOp:value:0*model/batchnorm_4/ReadVariableOp_1:value:09model/batchnorm_4/FusedBatchNormV3/ReadVariableOp:value:0;model/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
model/leaky_relu_4/LeakyRelu	LeakyRelu&model/batchnorm_4/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>v
model/conv_transpose_2/ShapeShape*model/leaky_relu_4/LeakyRelu:activations:0*
T0*
_output_shapes
:t
*model/conv_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,model/conv_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,model/conv_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$model/conv_transpose_2/strided_sliceStridedSlice%model/conv_transpose_2/Shape:output:03model/conv_transpose_2/strided_slice/stack:output:05model/conv_transpose_2/strided_slice/stack_1:output:05model/conv_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
model/conv_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`
model/conv_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :`
model/conv_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
model/conv_transpose_2/stackPack-model/conv_transpose_2/strided_slice:output:0'model/conv_transpose_2/stack/1:output:0'model/conv_transpose_2/stack/2:output:0'model/conv_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:v
,model/conv_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/conv_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/conv_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/conv_transpose_2/strided_slice_1StridedSlice%model/conv_transpose_2/stack:output:05model/conv_transpose_2/strided_slice_1/stack:output:07model/conv_transpose_2/strided_slice_1/stack_1:output:07model/conv_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
6model/conv_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp?model_conv_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
'model/conv_transpose_2/conv2d_transposeConv2DBackpropInput%model/conv_transpose_2/stack:output:0>model/conv_transpose_2/conv2d_transpose/ReadVariableOp:value:0*model/leaky_relu_4/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
-model/conv_transpose_2/BiasAdd/ReadVariableOpReadVariableOp6model_conv_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/conv_transpose_2/BiasAddBiasAdd0model/conv_transpose_2/conv2d_transpose:output:05model/conv_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
 model/batchnorm_5/ReadVariableOpReadVariableOp)model_batchnorm_5_readvariableop_resource*
_output_shapes
:*
dtype0?
"model/batchnorm_5/ReadVariableOp_1ReadVariableOp+model_batchnorm_5_readvariableop_1_resource*
_output_shapes
:*
dtype0?
1model/batchnorm_5/FusedBatchNormV3/ReadVariableOpReadVariableOp:model_batchnorm_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
3model/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<model_batchnorm_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
"model/batchnorm_5/FusedBatchNormV3FusedBatchNormV3'model/conv_transpose_2/BiasAdd:output:0(model/batchnorm_5/ReadVariableOp:value:0*model/batchnorm_5/ReadVariableOp_1:value:09model/batchnorm_5/FusedBatchNormV3/ReadVariableOp:value:0;model/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
model/leaky_relu_5/LeakyRelu	LeakyRelu&model/batchnorm_5/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>w
model/conv_transpose_30/ShapeShape*model/leaky_relu_5/LeakyRelu:activations:0*
T0*
_output_shapes
:u
+model/conv_transpose_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-model/conv_transpose_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-model/conv_transpose_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%model/conv_transpose_30/strided_sliceStridedSlice&model/conv_transpose_30/Shape:output:04model/conv_transpose_30/strided_slice/stack:output:06model/conv_transpose_30/strided_slice/stack_1:output:06model/conv_transpose_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model/conv_transpose_30/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@a
model/conv_transpose_30/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@a
model/conv_transpose_30/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
model/conv_transpose_30/stackPack.model/conv_transpose_30/strided_slice:output:0(model/conv_transpose_30/stack/1:output:0(model/conv_transpose_30/stack/2:output:0(model/conv_transpose_30/stack/3:output:0*
N*
T0*
_output_shapes
:w
-model/conv_transpose_30/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/model/conv_transpose_30/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/model/conv_transpose_30/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'model/conv_transpose_30/strided_slice_1StridedSlice&model/conv_transpose_30/stack:output:06model/conv_transpose_30/strided_slice_1/stack:output:08model/conv_transpose_30/strided_slice_1/stack_1:output:08model/conv_transpose_30/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7model/conv_transpose_30/conv2d_transpose/ReadVariableOpReadVariableOp@model_conv_transpose_30_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
(model/conv_transpose_30/conv2d_transposeConv2DBackpropInput&model/conv_transpose_30/stack:output:0?model/conv_transpose_30/conv2d_transpose/ReadVariableOp:value:0*model/leaky_relu_5/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
.model/conv_transpose_30/BiasAdd/ReadVariableOpReadVariableOp7model_conv_transpose_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/conv_transpose_30/BiasAddBiasAdd1model/conv_transpose_30/conv2d_transpose:output:06model/conv_transpose_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@?
!model/batchnorm_31/ReadVariableOpReadVariableOp*model_batchnorm_31_readvariableop_resource*
_output_shapes
:*
dtype0?
#model/batchnorm_31/ReadVariableOp_1ReadVariableOp,model_batchnorm_31_readvariableop_1_resource*
_output_shapes
:*
dtype0?
2model/batchnorm_31/FusedBatchNormV3/ReadVariableOpReadVariableOp;model_batchnorm_31_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
4model/batchnorm_31/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=model_batchnorm_31_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
#model/batchnorm_31/FusedBatchNormV3FusedBatchNormV3(model/conv_transpose_30/BiasAdd:output:0)model/batchnorm_31/ReadVariableOp:value:0+model/batchnorm_31/ReadVariableOp_1:value:0:model/batchnorm_31/FusedBatchNormV3/ReadVariableOp:value:0<model/batchnorm_31/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( ?
model/leaky_relu_32/LeakyRelu	LeakyRelu'model/batchnorm_31/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@*
alpha%???>w
model/conv_transpose_a/ShapeShape+model/leaky_relu_32/LeakyRelu:activations:0*
T0*
_output_shapes
:t
*model/conv_transpose_a/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,model/conv_transpose_a/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,model/conv_transpose_a/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$model/conv_transpose_a/strided_sliceStridedSlice%model/conv_transpose_a/Shape:output:03model/conv_transpose_a/strided_slice/stack:output:05model/conv_transpose_a/strided_slice/stack_1:output:05model/conv_transpose_a/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model/conv_transpose_a/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?a
model/conv_transpose_a/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?`
model/conv_transpose_a/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
model/conv_transpose_a/stackPack-model/conv_transpose_a/strided_slice:output:0'model/conv_transpose_a/stack/1:output:0'model/conv_transpose_a/stack/2:output:0'model/conv_transpose_a/stack/3:output:0*
N*
T0*
_output_shapes
:v
,model/conv_transpose_a/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/conv_transpose_a/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/conv_transpose_a/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/conv_transpose_a/strided_slice_1StridedSlice%model/conv_transpose_a/stack:output:05model/conv_transpose_a/strided_slice_1/stack:output:07model/conv_transpose_a/strided_slice_1/stack_1:output:07model/conv_transpose_a/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
6model/conv_transpose_a/conv2d_transpose/ReadVariableOpReadVariableOp?model_conv_transpose_a_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
'model/conv_transpose_a/conv2d_transposeConv2DBackpropInput%model/conv_transpose_a/stack:output:0>model/conv_transpose_a/conv2d_transpose/ReadVariableOp:value:0+model/leaky_relu_32/LeakyRelu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
-model/conv_transpose_a/BiasAdd/ReadVariableOpReadVariableOp6model_conv_transpose_a_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/conv_transpose_a/BiasAddBiasAdd0model/conv_transpose_a/conv2d_transpose:output:05model/conv_transpose_a/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
 model/batchnorm_b/ReadVariableOpReadVariableOp)model_batchnorm_b_readvariableop_resource*
_output_shapes
:*
dtype0?
"model/batchnorm_b/ReadVariableOp_1ReadVariableOp+model_batchnorm_b_readvariableop_1_resource*
_output_shapes
:*
dtype0?
1model/batchnorm_b/FusedBatchNormV3/ReadVariableOpReadVariableOp:model_batchnorm_b_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
3model/batchnorm_b/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<model_batchnorm_b_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
"model/batchnorm_b/FusedBatchNormV3FusedBatchNormV3'model/conv_transpose_a/BiasAdd:output:0(model/batchnorm_b/ReadVariableOp:value:0*model/batchnorm_b/ReadVariableOp_1:value:09model/batchnorm_b/FusedBatchNormV3/ReadVariableOp:value:0;model/batchnorm_b/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
model/leaky_relu_c/LeakyRelu	LeakyRelu&model/batchnorm_b/FusedBatchNormV3:y:0*1
_output_shapes
:???????????*
alpha%???>w
model/conv_transpose_4a/ShapeShape*model/leaky_relu_c/LeakyRelu:activations:0*
T0*
_output_shapes
:u
+model/conv_transpose_4a/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-model/conv_transpose_4a/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-model/conv_transpose_4a/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%model/conv_transpose_4a/strided_sliceStridedSlice&model/conv_transpose_4a/Shape:output:04model/conv_transpose_4a/strided_slice/stack:output:06model/conv_transpose_4a/strided_slice/stack_1:output:06model/conv_transpose_4a/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
model/conv_transpose_4a/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?b
model/conv_transpose_4a/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?a
model/conv_transpose_4a/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
model/conv_transpose_4a/stackPack.model/conv_transpose_4a/strided_slice:output:0(model/conv_transpose_4a/stack/1:output:0(model/conv_transpose_4a/stack/2:output:0(model/conv_transpose_4a/stack/3:output:0*
N*
T0*
_output_shapes
:w
-model/conv_transpose_4a/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/model/conv_transpose_4a/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/model/conv_transpose_4a/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'model/conv_transpose_4a/strided_slice_1StridedSlice&model/conv_transpose_4a/stack:output:06model/conv_transpose_4a/strided_slice_1/stack:output:08model/conv_transpose_4a/strided_slice_1/stack_1:output:08model/conv_transpose_4a/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
7model/conv_transpose_4a/conv2d_transpose/ReadVariableOpReadVariableOp@model_conv_transpose_4a_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
(model/conv_transpose_4a/conv2d_transposeConv2DBackpropInput&model/conv_transpose_4a/stack:output:0?model/conv_transpose_4a/conv2d_transpose/ReadVariableOp:value:0*model/leaky_relu_c/LeakyRelu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
.model/conv_transpose_4a/BiasAdd/ReadVariableOpReadVariableOp7model_conv_transpose_4a_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/conv_transpose_4a/BiasAddBiasAdd1model/conv_transpose_4a/conv2d_transpose:output:06model/conv_transpose_4a/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
model/conv_transpose_4a/SigmoidSigmoid(model/conv_transpose_4a/BiasAdd:output:0*
T0*1
_output_shapes
:???????????|
IdentityIdentity#model/conv_transpose_4a/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp2^model/batchnorm_1/FusedBatchNormV3/ReadVariableOp4^model/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1!^model/batchnorm_1/ReadVariableOp#^model/batchnorm_1/ReadVariableOp_12^model/batchnorm_2/FusedBatchNormV3/ReadVariableOp4^model/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1!^model/batchnorm_2/ReadVariableOp#^model/batchnorm_2/ReadVariableOp_12^model/batchnorm_3/FusedBatchNormV3/ReadVariableOp4^model/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1!^model/batchnorm_3/ReadVariableOp#^model/batchnorm_3/ReadVariableOp_13^model/batchnorm_31/FusedBatchNormV3/ReadVariableOp5^model/batchnorm_31/FusedBatchNormV3/ReadVariableOp_1"^model/batchnorm_31/ReadVariableOp$^model/batchnorm_31/ReadVariableOp_12^model/batchnorm_4/FusedBatchNormV3/ReadVariableOp4^model/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1!^model/batchnorm_4/ReadVariableOp#^model/batchnorm_4/ReadVariableOp_13^model/batchnorm_4a/FusedBatchNormV3/ReadVariableOp5^model/batchnorm_4a/FusedBatchNormV3/ReadVariableOp_1"^model/batchnorm_4a/ReadVariableOp$^model/batchnorm_4a/ReadVariableOp_12^model/batchnorm_5/FusedBatchNormV3/ReadVariableOp4^model/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1!^model/batchnorm_5/ReadVariableOp#^model/batchnorm_5/ReadVariableOp_12^model/batchnorm_b/FusedBatchNormV3/ReadVariableOp4^model/batchnorm_b/FusedBatchNormV3/ReadVariableOp_1!^model/batchnorm_b/ReadVariableOp#^model/batchnorm_b/ReadVariableOp_1$^model/conv_1/BiasAdd/ReadVariableOp#^model/conv_1/Conv2D/ReadVariableOp$^model/conv_2/BiasAdd/ReadVariableOp#^model/conv_2/Conv2D/ReadVariableOp$^model/conv_3/BiasAdd/ReadVariableOp#^model/conv_3/Conv2D/ReadVariableOp%^model/conv_4a/BiasAdd/ReadVariableOp$^model/conv_4a/Conv2D/ReadVariableOp.^model/conv_transpose_1/BiasAdd/ReadVariableOp7^model/conv_transpose_1/conv2d_transpose/ReadVariableOp.^model/conv_transpose_2/BiasAdd/ReadVariableOp7^model/conv_transpose_2/conv2d_transpose/ReadVariableOp/^model/conv_transpose_30/BiasAdd/ReadVariableOp8^model/conv_transpose_30/conv2d_transpose/ReadVariableOp/^model/conv_transpose_4a/BiasAdd/ReadVariableOp8^model/conv_transpose_4a/conv2d_transpose/ReadVariableOp.^model/conv_transpose_a/BiasAdd/ReadVariableOp7^model/conv_transpose_a/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2f
1model/batchnorm_1/FusedBatchNormV3/ReadVariableOp1model/batchnorm_1/FusedBatchNormV3/ReadVariableOp2j
3model/batchnorm_1/FusedBatchNormV3/ReadVariableOp_13model/batchnorm_1/FusedBatchNormV3/ReadVariableOp_12D
 model/batchnorm_1/ReadVariableOp model/batchnorm_1/ReadVariableOp2H
"model/batchnorm_1/ReadVariableOp_1"model/batchnorm_1/ReadVariableOp_12f
1model/batchnorm_2/FusedBatchNormV3/ReadVariableOp1model/batchnorm_2/FusedBatchNormV3/ReadVariableOp2j
3model/batchnorm_2/FusedBatchNormV3/ReadVariableOp_13model/batchnorm_2/FusedBatchNormV3/ReadVariableOp_12D
 model/batchnorm_2/ReadVariableOp model/batchnorm_2/ReadVariableOp2H
"model/batchnorm_2/ReadVariableOp_1"model/batchnorm_2/ReadVariableOp_12f
1model/batchnorm_3/FusedBatchNormV3/ReadVariableOp1model/batchnorm_3/FusedBatchNormV3/ReadVariableOp2j
3model/batchnorm_3/FusedBatchNormV3/ReadVariableOp_13model/batchnorm_3/FusedBatchNormV3/ReadVariableOp_12D
 model/batchnorm_3/ReadVariableOp model/batchnorm_3/ReadVariableOp2H
"model/batchnorm_3/ReadVariableOp_1"model/batchnorm_3/ReadVariableOp_12h
2model/batchnorm_31/FusedBatchNormV3/ReadVariableOp2model/batchnorm_31/FusedBatchNormV3/ReadVariableOp2l
4model/batchnorm_31/FusedBatchNormV3/ReadVariableOp_14model/batchnorm_31/FusedBatchNormV3/ReadVariableOp_12F
!model/batchnorm_31/ReadVariableOp!model/batchnorm_31/ReadVariableOp2J
#model/batchnorm_31/ReadVariableOp_1#model/batchnorm_31/ReadVariableOp_12f
1model/batchnorm_4/FusedBatchNormV3/ReadVariableOp1model/batchnorm_4/FusedBatchNormV3/ReadVariableOp2j
3model/batchnorm_4/FusedBatchNormV3/ReadVariableOp_13model/batchnorm_4/FusedBatchNormV3/ReadVariableOp_12D
 model/batchnorm_4/ReadVariableOp model/batchnorm_4/ReadVariableOp2H
"model/batchnorm_4/ReadVariableOp_1"model/batchnorm_4/ReadVariableOp_12h
2model/batchnorm_4a/FusedBatchNormV3/ReadVariableOp2model/batchnorm_4a/FusedBatchNormV3/ReadVariableOp2l
4model/batchnorm_4a/FusedBatchNormV3/ReadVariableOp_14model/batchnorm_4a/FusedBatchNormV3/ReadVariableOp_12F
!model/batchnorm_4a/ReadVariableOp!model/batchnorm_4a/ReadVariableOp2J
#model/batchnorm_4a/ReadVariableOp_1#model/batchnorm_4a/ReadVariableOp_12f
1model/batchnorm_5/FusedBatchNormV3/ReadVariableOp1model/batchnorm_5/FusedBatchNormV3/ReadVariableOp2j
3model/batchnorm_5/FusedBatchNormV3/ReadVariableOp_13model/batchnorm_5/FusedBatchNormV3/ReadVariableOp_12D
 model/batchnorm_5/ReadVariableOp model/batchnorm_5/ReadVariableOp2H
"model/batchnorm_5/ReadVariableOp_1"model/batchnorm_5/ReadVariableOp_12f
1model/batchnorm_b/FusedBatchNormV3/ReadVariableOp1model/batchnorm_b/FusedBatchNormV3/ReadVariableOp2j
3model/batchnorm_b/FusedBatchNormV3/ReadVariableOp_13model/batchnorm_b/FusedBatchNormV3/ReadVariableOp_12D
 model/batchnorm_b/ReadVariableOp model/batchnorm_b/ReadVariableOp2H
"model/batchnorm_b/ReadVariableOp_1"model/batchnorm_b/ReadVariableOp_12J
#model/conv_1/BiasAdd/ReadVariableOp#model/conv_1/BiasAdd/ReadVariableOp2H
"model/conv_1/Conv2D/ReadVariableOp"model/conv_1/Conv2D/ReadVariableOp2J
#model/conv_2/BiasAdd/ReadVariableOp#model/conv_2/BiasAdd/ReadVariableOp2H
"model/conv_2/Conv2D/ReadVariableOp"model/conv_2/Conv2D/ReadVariableOp2J
#model/conv_3/BiasAdd/ReadVariableOp#model/conv_3/BiasAdd/ReadVariableOp2H
"model/conv_3/Conv2D/ReadVariableOp"model/conv_3/Conv2D/ReadVariableOp2L
$model/conv_4a/BiasAdd/ReadVariableOp$model/conv_4a/BiasAdd/ReadVariableOp2J
#model/conv_4a/Conv2D/ReadVariableOp#model/conv_4a/Conv2D/ReadVariableOp2^
-model/conv_transpose_1/BiasAdd/ReadVariableOp-model/conv_transpose_1/BiasAdd/ReadVariableOp2p
6model/conv_transpose_1/conv2d_transpose/ReadVariableOp6model/conv_transpose_1/conv2d_transpose/ReadVariableOp2^
-model/conv_transpose_2/BiasAdd/ReadVariableOp-model/conv_transpose_2/BiasAdd/ReadVariableOp2p
6model/conv_transpose_2/conv2d_transpose/ReadVariableOp6model/conv_transpose_2/conv2d_transpose/ReadVariableOp2`
.model/conv_transpose_30/BiasAdd/ReadVariableOp.model/conv_transpose_30/BiasAdd/ReadVariableOp2r
7model/conv_transpose_30/conv2d_transpose/ReadVariableOp7model/conv_transpose_30/conv2d_transpose/ReadVariableOp2`
.model/conv_transpose_4a/BiasAdd/ReadVariableOp.model/conv_transpose_4a/BiasAdd/ReadVariableOp2r
7model/conv_transpose_4a/conv2d_transpose/ReadVariableOp7model/conv_transpose_4a/conv2d_transpose/ReadVariableOp2^
-model/conv_transpose_a/BiasAdd/ReadVariableOp-model/conv_transpose_a/BiasAdd/ReadVariableOp2p
6model/conv_transpose_a/conv2d_transpose/ReadVariableOp6model/conv_transpose_a/conv2d_transpose/ReadVariableOp:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameinput_layer
?
b
F__inference_leaky_relu_3_layer_call_and_return_conditional_losses_4421

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?(
?__inference_model_layer_call_and_return_conditional_losses_3797

inputs?
%conv_1_conv2d_readvariableop_resource:4
&conv_1_biasadd_readvariableop_resource:1
#batchnorm_1_readvariableop_resource:3
%batchnorm_1_readvariableop_1_resource:B
4batchnorm_1_fusedbatchnormv3_readvariableop_resource:D
6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource:?
%conv_2_conv2d_readvariableop_resource:4
&conv_2_biasadd_readvariableop_resource:1
#batchnorm_2_readvariableop_resource:3
%batchnorm_2_readvariableop_1_resource:B
4batchnorm_2_fusedbatchnormv3_readvariableop_resource:D
6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource:?
%conv_3_conv2d_readvariableop_resource:4
&conv_3_biasadd_readvariableop_resource:1
#batchnorm_3_readvariableop_resource:3
%batchnorm_3_readvariableop_1_resource:B
4batchnorm_3_fusedbatchnormv3_readvariableop_resource:D
6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource:@
&conv_4a_conv2d_readvariableop_resource:5
'conv_4a_biasadd_readvariableop_resource:2
$batchnorm_4a_readvariableop_resource:4
&batchnorm_4a_readvariableop_1_resource:C
5batchnorm_4a_fusedbatchnormv3_readvariableop_resource:E
7batchnorm_4a_fusedbatchnormv3_readvariableop_1_resource:S
9conv_transpose_1_conv2d_transpose_readvariableop_resource:>
0conv_transpose_1_biasadd_readvariableop_resource:1
#batchnorm_4_readvariableop_resource:3
%batchnorm_4_readvariableop_1_resource:B
4batchnorm_4_fusedbatchnormv3_readvariableop_resource:D
6batchnorm_4_fusedbatchnormv3_readvariableop_1_resource:S
9conv_transpose_2_conv2d_transpose_readvariableop_resource:>
0conv_transpose_2_biasadd_readvariableop_resource:1
#batchnorm_5_readvariableop_resource:3
%batchnorm_5_readvariableop_1_resource:B
4batchnorm_5_fusedbatchnormv3_readvariableop_resource:D
6batchnorm_5_fusedbatchnormv3_readvariableop_1_resource:T
:conv_transpose_30_conv2d_transpose_readvariableop_resource:?
1conv_transpose_30_biasadd_readvariableop_resource:2
$batchnorm_31_readvariableop_resource:4
&batchnorm_31_readvariableop_1_resource:C
5batchnorm_31_fusedbatchnormv3_readvariableop_resource:E
7batchnorm_31_fusedbatchnormv3_readvariableop_1_resource:S
9conv_transpose_a_conv2d_transpose_readvariableop_resource:>
0conv_transpose_a_biasadd_readvariableop_resource:1
#batchnorm_b_readvariableop_resource:3
%batchnorm_b_readvariableop_1_resource:B
4batchnorm_b_fusedbatchnormv3_readvariableop_resource:D
6batchnorm_b_fusedbatchnormv3_readvariableop_1_resource:T
:conv_transpose_4a_conv2d_transpose_readvariableop_resource:?
1conv_transpose_4a_biasadd_readvariableop_resource:
identity??+batchnorm_1/FusedBatchNormV3/ReadVariableOp?-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?batchnorm_1/ReadVariableOp?batchnorm_1/ReadVariableOp_1?+batchnorm_2/FusedBatchNormV3/ReadVariableOp?-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1?batchnorm_2/ReadVariableOp?batchnorm_2/ReadVariableOp_1?+batchnorm_3/FusedBatchNormV3/ReadVariableOp?-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1?batchnorm_3/ReadVariableOp?batchnorm_3/ReadVariableOp_1?,batchnorm_31/FusedBatchNormV3/ReadVariableOp?.batchnorm_31/FusedBatchNormV3/ReadVariableOp_1?batchnorm_31/ReadVariableOp?batchnorm_31/ReadVariableOp_1?+batchnorm_4/FusedBatchNormV3/ReadVariableOp?-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1?batchnorm_4/ReadVariableOp?batchnorm_4/ReadVariableOp_1?,batchnorm_4a/FusedBatchNormV3/ReadVariableOp?.batchnorm_4a/FusedBatchNormV3/ReadVariableOp_1?batchnorm_4a/ReadVariableOp?batchnorm_4a/ReadVariableOp_1?+batchnorm_5/FusedBatchNormV3/ReadVariableOp?-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1?batchnorm_5/ReadVariableOp?batchnorm_5/ReadVariableOp_1?+batchnorm_b/FusedBatchNormV3/ReadVariableOp?-batchnorm_b/FusedBatchNormV3/ReadVariableOp_1?batchnorm_b/ReadVariableOp?batchnorm_b/ReadVariableOp_1?conv_1/BiasAdd/ReadVariableOp?conv_1/Conv2D/ReadVariableOp?conv_2/BiasAdd/ReadVariableOp?conv_2/Conv2D/ReadVariableOp?conv_3/BiasAdd/ReadVariableOp?conv_3/Conv2D/ReadVariableOp?conv_4a/BiasAdd/ReadVariableOp?conv_4a/Conv2D/ReadVariableOp?'conv_transpose_1/BiasAdd/ReadVariableOp?0conv_transpose_1/conv2d_transpose/ReadVariableOp?'conv_transpose_2/BiasAdd/ReadVariableOp?0conv_transpose_2/conv2d_transpose/ReadVariableOp?(conv_transpose_30/BiasAdd/ReadVariableOp?1conv_transpose_30/conv2d_transpose/ReadVariableOp?(conv_transpose_4a/BiasAdd/ReadVariableOp?1conv_transpose_4a/conv2d_transpose/ReadVariableOp?'conv_transpose_a/BiasAdd/ReadVariableOp?0conv_transpose_a/conv2d_transpose/ReadVariableOp?
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv_1/Conv2DConv2Dinputs$conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????z
batchnorm_1/ReadVariableOpReadVariableOp#batchnorm_1_readvariableop_resource*
_output_shapes
:*
dtype0~
batchnorm_1/ReadVariableOp_1ReadVariableOp%batchnorm_1_readvariableop_1_resource*
_output_shapes
:*
dtype0?
+batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
batchnorm_1/FusedBatchNormV3FusedBatchNormV3conv_1/BiasAdd:output:0"batchnorm_1/ReadVariableOp:value:0$batchnorm_1/ReadVariableOp_1:value:03batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
leaky_relu_1/LeakyRelu	LeakyRelu batchnorm_1/FusedBatchNormV3:y:0*1
_output_shapes
:???????????*
alpha%???>?
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv_2/Conv2DConv2D$leaky_relu_1/LeakyRelu:activations:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
?
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  z
batchnorm_2/ReadVariableOpReadVariableOp#batchnorm_2_readvariableop_resource*
_output_shapes
:*
dtype0~
batchnorm_2/ReadVariableOp_1ReadVariableOp%batchnorm_2_readvariableop_1_resource*
_output_shapes
:*
dtype0?
+batchnorm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
batchnorm_2/FusedBatchNormV3FusedBatchNormV3conv_2/BiasAdd:output:0"batchnorm_2/ReadVariableOp:value:0$batchnorm_2/ReadVariableOp_1:value:03batchnorm_2/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( ?
leaky_relu_2/LeakyRelu	LeakyRelu batchnorm_2/FusedBatchNormV3:y:0*/
_output_shapes
:?????????  *
alpha%???>?
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv_3/Conv2DConv2D$leaky_relu_2/LeakyRelu:activations:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z
batchnorm_3/ReadVariableOpReadVariableOp#batchnorm_3_readvariableop_resource*
_output_shapes
:*
dtype0~
batchnorm_3/ReadVariableOp_1ReadVariableOp%batchnorm_3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
+batchnorm_3/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
batchnorm_3/FusedBatchNormV3FusedBatchNormV3conv_3/BiasAdd:output:0"batchnorm_3/ReadVariableOp:value:0$batchnorm_3/ReadVariableOp_1:value:03batchnorm_3/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
leaky_relu_3/LeakyRelu	LeakyRelu batchnorm_3/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>?
conv_4a/Conv2D/ReadVariableOpReadVariableOp&conv_4a_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv_4a/Conv2DConv2D$leaky_relu_3/LeakyRelu:activations:0%conv_4a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv_4a/BiasAdd/ReadVariableOpReadVariableOp'conv_4a_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_4a/BiasAddBiasAddconv_4a/Conv2D:output:0&conv_4a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????|
batchnorm_4a/ReadVariableOpReadVariableOp$batchnorm_4a_readvariableop_resource*
_output_shapes
:*
dtype0?
batchnorm_4a/ReadVariableOp_1ReadVariableOp&batchnorm_4a_readvariableop_1_resource*
_output_shapes
:*
dtype0?
,batchnorm_4a/FusedBatchNormV3/ReadVariableOpReadVariableOp5batchnorm_4a_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
.batchnorm_4a/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7batchnorm_4a_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
batchnorm_4a/FusedBatchNormV3FusedBatchNormV3conv_4a/BiasAdd:output:0#batchnorm_4a/ReadVariableOp:value:0%batchnorm_4a/ReadVariableOp_1:value:04batchnorm_4a/FusedBatchNormV3/ReadVariableOp:value:06batchnorm_4a/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
leaky_relu_4a/LeakyRelu	LeakyRelu!batchnorm_4a/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>k
conv_transpose_1/ShapeShape%leaky_relu_4a/LeakyRelu:activations:0*
T0*
_output_shapes
:n
$conv_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv_transpose_1/strided_sliceStridedSliceconv_transpose_1/Shape:output:0-conv_transpose_1/strided_slice/stack:output:0/conv_transpose_1/strided_slice/stack_1:output:0/conv_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv_transpose_1/stackPack'conv_transpose_1/strided_slice:output:0!conv_transpose_1/stack/1:output:0!conv_transpose_1/stack/2:output:0!conv_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv_transpose_1/strided_slice_1StridedSliceconv_transpose_1/stack:output:0/conv_transpose_1/strided_slice_1/stack:output:01conv_transpose_1/strided_slice_1/stack_1:output:01conv_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
!conv_transpose_1/conv2d_transposeConv2DBackpropInputconv_transpose_1/stack:output:08conv_transpose_1/conv2d_transpose/ReadVariableOp:value:0%leaky_relu_4a/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
'conv_transpose_1/BiasAdd/ReadVariableOpReadVariableOp0conv_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_transpose_1/BiasAddBiasAdd*conv_transpose_1/conv2d_transpose:output:0/conv_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z
batchnorm_4/ReadVariableOpReadVariableOp#batchnorm_4_readvariableop_resource*
_output_shapes
:*
dtype0~
batchnorm_4/ReadVariableOp_1ReadVariableOp%batchnorm_4_readvariableop_1_resource*
_output_shapes
:*
dtype0?
+batchnorm_4/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
batchnorm_4/FusedBatchNormV3FusedBatchNormV3!conv_transpose_1/BiasAdd:output:0"batchnorm_4/ReadVariableOp:value:0$batchnorm_4/ReadVariableOp_1:value:03batchnorm_4/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
leaky_relu_4/LeakyRelu	LeakyRelu batchnorm_4/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>j
conv_transpose_2/ShapeShape$leaky_relu_4/LeakyRelu:activations:0*
T0*
_output_shapes
:n
$conv_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv_transpose_2/strided_sliceStridedSliceconv_transpose_2/Shape:output:0-conv_transpose_2/strided_slice/stack:output:0/conv_transpose_2/strided_slice/stack_1:output:0/conv_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv_transpose_2/stackPack'conv_transpose_2/strided_slice:output:0!conv_transpose_2/stack/1:output:0!conv_transpose_2/stack/2:output:0!conv_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv_transpose_2/strided_slice_1StridedSliceconv_transpose_2/stack:output:0/conv_transpose_2/strided_slice_1/stack:output:01conv_transpose_2/strided_slice_1/stack_1:output:01conv_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
!conv_transpose_2/conv2d_transposeConv2DBackpropInputconv_transpose_2/stack:output:08conv_transpose_2/conv2d_transpose/ReadVariableOp:value:0$leaky_relu_4/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
'conv_transpose_2/BiasAdd/ReadVariableOpReadVariableOp0conv_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_transpose_2/BiasAddBiasAdd*conv_transpose_2/conv2d_transpose:output:0/conv_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z
batchnorm_5/ReadVariableOpReadVariableOp#batchnorm_5_readvariableop_resource*
_output_shapes
:*
dtype0~
batchnorm_5/ReadVariableOp_1ReadVariableOp%batchnorm_5_readvariableop_1_resource*
_output_shapes
:*
dtype0?
+batchnorm_5/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
batchnorm_5/FusedBatchNormV3FusedBatchNormV3!conv_transpose_2/BiasAdd:output:0"batchnorm_5/ReadVariableOp:value:0$batchnorm_5/ReadVariableOp_1:value:03batchnorm_5/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
leaky_relu_5/LeakyRelu	LeakyRelu batchnorm_5/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
alpha%???>k
conv_transpose_30/ShapeShape$leaky_relu_5/LeakyRelu:activations:0*
T0*
_output_shapes
:o
%conv_transpose_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'conv_transpose_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'conv_transpose_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv_transpose_30/strided_sliceStridedSlice conv_transpose_30/Shape:output:0.conv_transpose_30/strided_slice/stack:output:00conv_transpose_30/strided_slice/stack_1:output:00conv_transpose_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
conv_transpose_30/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@[
conv_transpose_30/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@[
conv_transpose_30/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv_transpose_30/stackPack(conv_transpose_30/strided_slice:output:0"conv_transpose_30/stack/1:output:0"conv_transpose_30/stack/2:output:0"conv_transpose_30/stack/3:output:0*
N*
T0*
_output_shapes
:q
'conv_transpose_30/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv_transpose_30/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv_transpose_30/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv_transpose_30/strided_slice_1StridedSlice conv_transpose_30/stack:output:00conv_transpose_30/strided_slice_1/stack:output:02conv_transpose_30/strided_slice_1/stack_1:output:02conv_transpose_30/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1conv_transpose_30/conv2d_transpose/ReadVariableOpReadVariableOp:conv_transpose_30_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
"conv_transpose_30/conv2d_transposeConv2DBackpropInput conv_transpose_30/stack:output:09conv_transpose_30/conv2d_transpose/ReadVariableOp:value:0$leaky_relu_5/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
(conv_transpose_30/BiasAdd/ReadVariableOpReadVariableOp1conv_transpose_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_transpose_30/BiasAddBiasAdd+conv_transpose_30/conv2d_transpose:output:00conv_transpose_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@|
batchnorm_31/ReadVariableOpReadVariableOp$batchnorm_31_readvariableop_resource*
_output_shapes
:*
dtype0?
batchnorm_31/ReadVariableOp_1ReadVariableOp&batchnorm_31_readvariableop_1_resource*
_output_shapes
:*
dtype0?
,batchnorm_31/FusedBatchNormV3/ReadVariableOpReadVariableOp5batchnorm_31_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
.batchnorm_31/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7batchnorm_31_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
batchnorm_31/FusedBatchNormV3FusedBatchNormV3"conv_transpose_30/BiasAdd:output:0#batchnorm_31/ReadVariableOp:value:0%batchnorm_31/ReadVariableOp_1:value:04batchnorm_31/FusedBatchNormV3/ReadVariableOp:value:06batchnorm_31/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( ?
leaky_relu_32/LeakyRelu	LeakyRelu!batchnorm_31/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@*
alpha%???>k
conv_transpose_a/ShapeShape%leaky_relu_32/LeakyRelu:activations:0*
T0*
_output_shapes
:n
$conv_transpose_a/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv_transpose_a/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv_transpose_a/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv_transpose_a/strided_sliceStridedSliceconv_transpose_a/Shape:output:0-conv_transpose_a/strided_slice/stack:output:0/conv_transpose_a/strided_slice/stack_1:output:0/conv_transpose_a/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
conv_transpose_a/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?[
conv_transpose_a/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?Z
conv_transpose_a/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv_transpose_a/stackPack'conv_transpose_a/strided_slice:output:0!conv_transpose_a/stack/1:output:0!conv_transpose_a/stack/2:output:0!conv_transpose_a/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv_transpose_a/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv_transpose_a/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv_transpose_a/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv_transpose_a/strided_slice_1StridedSliceconv_transpose_a/stack:output:0/conv_transpose_a/strided_slice_1/stack:output:01conv_transpose_a/strided_slice_1/stack_1:output:01conv_transpose_a/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv_transpose_a/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_a_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
!conv_transpose_a/conv2d_transposeConv2DBackpropInputconv_transpose_a/stack:output:08conv_transpose_a/conv2d_transpose/ReadVariableOp:value:0%leaky_relu_32/LeakyRelu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
'conv_transpose_a/BiasAdd/ReadVariableOpReadVariableOp0conv_transpose_a_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_transpose_a/BiasAddBiasAdd*conv_transpose_a/conv2d_transpose:output:0/conv_transpose_a/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????z
batchnorm_b/ReadVariableOpReadVariableOp#batchnorm_b_readvariableop_resource*
_output_shapes
:*
dtype0~
batchnorm_b/ReadVariableOp_1ReadVariableOp%batchnorm_b_readvariableop_1_resource*
_output_shapes
:*
dtype0?
+batchnorm_b/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_b_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
-batchnorm_b/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_b_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
batchnorm_b/FusedBatchNormV3FusedBatchNormV3!conv_transpose_a/BiasAdd:output:0"batchnorm_b/ReadVariableOp:value:0$batchnorm_b/ReadVariableOp_1:value:03batchnorm_b/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_b/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
leaky_relu_c/LeakyRelu	LeakyRelu batchnorm_b/FusedBatchNormV3:y:0*1
_output_shapes
:???????????*
alpha%???>k
conv_transpose_4a/ShapeShape$leaky_relu_c/LeakyRelu:activations:0*
T0*
_output_shapes
:o
%conv_transpose_4a/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'conv_transpose_4a/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'conv_transpose_4a/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv_transpose_4a/strided_sliceStridedSlice conv_transpose_4a/Shape:output:0.conv_transpose_4a/strided_slice/stack:output:00conv_transpose_4a/strided_slice/stack_1:output:00conv_transpose_4a/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv_transpose_4a/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?\
conv_transpose_4a/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?[
conv_transpose_4a/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv_transpose_4a/stackPack(conv_transpose_4a/strided_slice:output:0"conv_transpose_4a/stack/1:output:0"conv_transpose_4a/stack/2:output:0"conv_transpose_4a/stack/3:output:0*
N*
T0*
_output_shapes
:q
'conv_transpose_4a/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv_transpose_4a/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv_transpose_4a/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv_transpose_4a/strided_slice_1StridedSlice conv_transpose_4a/stack:output:00conv_transpose_4a/strided_slice_1/stack:output:02conv_transpose_4a/strided_slice_1/stack_1:output:02conv_transpose_4a/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1conv_transpose_4a/conv2d_transpose/ReadVariableOpReadVariableOp:conv_transpose_4a_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
"conv_transpose_4a/conv2d_transposeConv2DBackpropInput conv_transpose_4a/stack:output:09conv_transpose_4a/conv2d_transpose/ReadVariableOp:value:0$leaky_relu_c/LeakyRelu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
(conv_transpose_4a/BiasAdd/ReadVariableOpReadVariableOp1conv_transpose_4a_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv_transpose_4a/BiasAddBiasAdd+conv_transpose_4a/conv2d_transpose:output:00conv_transpose_4a/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
conv_transpose_4a/SigmoidSigmoid"conv_transpose_4a/BiasAdd:output:0*
T0*1
_output_shapes
:???????????v
IdentityIdentityconv_transpose_4a/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp,^batchnorm_1/FusedBatchNormV3/ReadVariableOp.^batchnorm_1/FusedBatchNormV3/ReadVariableOp_1^batchnorm_1/ReadVariableOp^batchnorm_1/ReadVariableOp_1,^batchnorm_2/FusedBatchNormV3/ReadVariableOp.^batchnorm_2/FusedBatchNormV3/ReadVariableOp_1^batchnorm_2/ReadVariableOp^batchnorm_2/ReadVariableOp_1,^batchnorm_3/FusedBatchNormV3/ReadVariableOp.^batchnorm_3/FusedBatchNormV3/ReadVariableOp_1^batchnorm_3/ReadVariableOp^batchnorm_3/ReadVariableOp_1-^batchnorm_31/FusedBatchNormV3/ReadVariableOp/^batchnorm_31/FusedBatchNormV3/ReadVariableOp_1^batchnorm_31/ReadVariableOp^batchnorm_31/ReadVariableOp_1,^batchnorm_4/FusedBatchNormV3/ReadVariableOp.^batchnorm_4/FusedBatchNormV3/ReadVariableOp_1^batchnorm_4/ReadVariableOp^batchnorm_4/ReadVariableOp_1-^batchnorm_4a/FusedBatchNormV3/ReadVariableOp/^batchnorm_4a/FusedBatchNormV3/ReadVariableOp_1^batchnorm_4a/ReadVariableOp^batchnorm_4a/ReadVariableOp_1,^batchnorm_5/FusedBatchNormV3/ReadVariableOp.^batchnorm_5/FusedBatchNormV3/ReadVariableOp_1^batchnorm_5/ReadVariableOp^batchnorm_5/ReadVariableOp_1,^batchnorm_b/FusedBatchNormV3/ReadVariableOp.^batchnorm_b/FusedBatchNormV3/ReadVariableOp_1^batchnorm_b/ReadVariableOp^batchnorm_b/ReadVariableOp_1^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^conv_4a/BiasAdd/ReadVariableOp^conv_4a/Conv2D/ReadVariableOp(^conv_transpose_1/BiasAdd/ReadVariableOp1^conv_transpose_1/conv2d_transpose/ReadVariableOp(^conv_transpose_2/BiasAdd/ReadVariableOp1^conv_transpose_2/conv2d_transpose/ReadVariableOp)^conv_transpose_30/BiasAdd/ReadVariableOp2^conv_transpose_30/conv2d_transpose/ReadVariableOp)^conv_transpose_4a/BiasAdd/ReadVariableOp2^conv_transpose_4a/conv2d_transpose/ReadVariableOp(^conv_transpose_a/BiasAdd/ReadVariableOp1^conv_transpose_a/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batchnorm_1/FusedBatchNormV3/ReadVariableOp+batchnorm_1/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1-batchnorm_1/FusedBatchNormV3/ReadVariableOp_128
batchnorm_1/ReadVariableOpbatchnorm_1/ReadVariableOp2<
batchnorm_1/ReadVariableOp_1batchnorm_1/ReadVariableOp_12Z
+batchnorm_2/FusedBatchNormV3/ReadVariableOp+batchnorm_2/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1-batchnorm_2/FusedBatchNormV3/ReadVariableOp_128
batchnorm_2/ReadVariableOpbatchnorm_2/ReadVariableOp2<
batchnorm_2/ReadVariableOp_1batchnorm_2/ReadVariableOp_12Z
+batchnorm_3/FusedBatchNormV3/ReadVariableOp+batchnorm_3/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1-batchnorm_3/FusedBatchNormV3/ReadVariableOp_128
batchnorm_3/ReadVariableOpbatchnorm_3/ReadVariableOp2<
batchnorm_3/ReadVariableOp_1batchnorm_3/ReadVariableOp_12\
,batchnorm_31/FusedBatchNormV3/ReadVariableOp,batchnorm_31/FusedBatchNormV3/ReadVariableOp2`
.batchnorm_31/FusedBatchNormV3/ReadVariableOp_1.batchnorm_31/FusedBatchNormV3/ReadVariableOp_12:
batchnorm_31/ReadVariableOpbatchnorm_31/ReadVariableOp2>
batchnorm_31/ReadVariableOp_1batchnorm_31/ReadVariableOp_12Z
+batchnorm_4/FusedBatchNormV3/ReadVariableOp+batchnorm_4/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1-batchnorm_4/FusedBatchNormV3/ReadVariableOp_128
batchnorm_4/ReadVariableOpbatchnorm_4/ReadVariableOp2<
batchnorm_4/ReadVariableOp_1batchnorm_4/ReadVariableOp_12\
,batchnorm_4a/FusedBatchNormV3/ReadVariableOp,batchnorm_4a/FusedBatchNormV3/ReadVariableOp2`
.batchnorm_4a/FusedBatchNormV3/ReadVariableOp_1.batchnorm_4a/FusedBatchNormV3/ReadVariableOp_12:
batchnorm_4a/ReadVariableOpbatchnorm_4a/ReadVariableOp2>
batchnorm_4a/ReadVariableOp_1batchnorm_4a/ReadVariableOp_12Z
+batchnorm_5/FusedBatchNormV3/ReadVariableOp+batchnorm_5/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1-batchnorm_5/FusedBatchNormV3/ReadVariableOp_128
batchnorm_5/ReadVariableOpbatchnorm_5/ReadVariableOp2<
batchnorm_5/ReadVariableOp_1batchnorm_5/ReadVariableOp_12Z
+batchnorm_b/FusedBatchNormV3/ReadVariableOp+batchnorm_b/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_b/FusedBatchNormV3/ReadVariableOp_1-batchnorm_b/FusedBatchNormV3/ReadVariableOp_128
batchnorm_b/ReadVariableOpbatchnorm_b/ReadVariableOp2<
batchnorm_b/ReadVariableOp_1batchnorm_b/ReadVariableOp_12>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp2@
conv_4a/BiasAdd/ReadVariableOpconv_4a/BiasAdd/ReadVariableOp2>
conv_4a/Conv2D/ReadVariableOpconv_4a/Conv2D/ReadVariableOp2R
'conv_transpose_1/BiasAdd/ReadVariableOp'conv_transpose_1/BiasAdd/ReadVariableOp2d
0conv_transpose_1/conv2d_transpose/ReadVariableOp0conv_transpose_1/conv2d_transpose/ReadVariableOp2R
'conv_transpose_2/BiasAdd/ReadVariableOp'conv_transpose_2/BiasAdd/ReadVariableOp2d
0conv_transpose_2/conv2d_transpose/ReadVariableOp0conv_transpose_2/conv2d_transpose/ReadVariableOp2T
(conv_transpose_30/BiasAdd/ReadVariableOp(conv_transpose_30/BiasAdd/ReadVariableOp2f
1conv_transpose_30/conv2d_transpose/ReadVariableOp1conv_transpose_30/conv2d_transpose/ReadVariableOp2T
(conv_transpose_4a/BiasAdd/ReadVariableOp(conv_transpose_4a/BiasAdd/ReadVariableOp2f
1conv_transpose_4a/conv2d_transpose/ReadVariableOp1conv_transpose_4a/conv2d_transpose/ReadVariableOp2R
'conv_transpose_a/BiasAdd/ReadVariableOp'conv_transpose_a/BiasAdd/ReadVariableOp2d
0conv_transpose_a/conv2d_transpose/ReadVariableOp0conv_transpose_a/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?x
?
?__inference_model_layer_call_and_return_conditional_losses_3214
input_layer%
conv_1_3088:
conv_1_3090:
batchnorm_1_3093:
batchnorm_1_3095:
batchnorm_1_3097:
batchnorm_1_3099:%
conv_2_3103:
conv_2_3105:
batchnorm_2_3108:
batchnorm_2_3110:
batchnorm_2_3112:
batchnorm_2_3114:%
conv_3_3118:
conv_3_3120:
batchnorm_3_3123:
batchnorm_3_3125:
batchnorm_3_3127:
batchnorm_3_3129:&
conv_4a_3133:
conv_4a_3135:
batchnorm_4a_3138:
batchnorm_4a_3140:
batchnorm_4a_3142:
batchnorm_4a_3144:/
conv_transpose_1_3148:#
conv_transpose_1_3150:
batchnorm_4_3153:
batchnorm_4_3155:
batchnorm_4_3157:
batchnorm_4_3159:/
conv_transpose_2_3163:#
conv_transpose_2_3165:
batchnorm_5_3168:
batchnorm_5_3170:
batchnorm_5_3172:
batchnorm_5_3174:0
conv_transpose_30_3178:$
conv_transpose_30_3180:
batchnorm_31_3183:
batchnorm_31_3185:
batchnorm_31_3187:
batchnorm_31_3189:/
conv_transpose_a_3193:#
conv_transpose_a_3195:
batchnorm_b_3198:
batchnorm_b_3200:
batchnorm_b_3202:
batchnorm_b_3204:0
conv_transpose_4a_3208:$
conv_transpose_4a_3210:
identity??#batchnorm_1/StatefulPartitionedCall?#batchnorm_2/StatefulPartitionedCall?#batchnorm_3/StatefulPartitionedCall?$batchnorm_31/StatefulPartitionedCall?#batchnorm_4/StatefulPartitionedCall?$batchnorm_4a/StatefulPartitionedCall?#batchnorm_5/StatefulPartitionedCall?#batchnorm_b/StatefulPartitionedCall?conv_1/StatefulPartitionedCall?conv_2/StatefulPartitionedCall?conv_3/StatefulPartitionedCall?conv_4a/StatefulPartitionedCall?(conv_transpose_1/StatefulPartitionedCall?(conv_transpose_2/StatefulPartitionedCall?)conv_transpose_30/StatefulPartitionedCall?)conv_transpose_4a/StatefulPartitionedCall?(conv_transpose_a/StatefulPartitionedCall?
conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv_1_3088conv_1_3090*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_1_layer_call_and_return_conditional_losses_2242?
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batchnorm_1_3093batchnorm_1_3095batchnorm_1_3097batchnorm_1_3099*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_1514?
leaky_relu_1/PartitionedCallPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_1_layer_call_and_return_conditional_losses_2262?
conv_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_1/PartitionedCall:output:0conv_2_3103conv_2_3105*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_2_layer_call_and_return_conditional_losses_2274?
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batchnorm_2_3108batchnorm_2_3110batchnorm_2_3112batchnorm_2_3114*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_1578?
leaky_relu_2/PartitionedCallPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_2_layer_call_and_return_conditional_losses_2294?
conv_3/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_2/PartitionedCall:output:0conv_3_3118conv_3_3120*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_3_layer_call_and_return_conditional_losses_2306?
#batchnorm_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batchnorm_3_3123batchnorm_3_3125batchnorm_3_3127batchnorm_3_3129*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_1642?
leaky_relu_3/PartitionedCallPartitionedCall,batchnorm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_3_layer_call_and_return_conditional_losses_2326?
conv_4a/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_3/PartitionedCall:output:0conv_4a_3133conv_4a_3135*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv_4a_layer_call_and_return_conditional_losses_2338?
$batchnorm_4a/StatefulPartitionedCallStatefulPartitionedCall(conv_4a/StatefulPartitionedCall:output:0batchnorm_4a_3138batchnorm_4a_3140batchnorm_4a_3142batchnorm_4a_3144*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_batchnorm_4a_layer_call_and_return_conditional_losses_1706?
leaky_relu_4a/PartitionedCallPartitionedCall-batchnorm_4a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_leaky_relu_4a_layer_call_and_return_conditional_losses_2358?
(conv_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_relu_4a/PartitionedCall:output:0conv_transpose_1_3148conv_transpose_1_3150*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_1_layer_call_and_return_conditional_losses_1785?
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_1/StatefulPartitionedCall:output:0batchnorm_4_3153batchnorm_4_3155batchnorm_4_3157batchnorm_4_3159*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_1814?
leaky_relu_4/PartitionedCallPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_4_layer_call_and_return_conditional_losses_2379?
(conv_transpose_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_4/PartitionedCall:output:0conv_transpose_2_3163conv_transpose_2_3165*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_2_layer_call_and_return_conditional_losses_1893?
#batchnorm_5/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_2/StatefulPartitionedCall:output:0batchnorm_5_3168batchnorm_5_3170batchnorm_5_3172batchnorm_5_3174*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_1922?
leaky_relu_5/PartitionedCallPartitionedCall,batchnorm_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_5_layer_call_and_return_conditional_losses_2400?
)conv_transpose_30/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_5/PartitionedCall:output:0conv_transpose_30_3178conv_transpose_30_3180*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv_transpose_30_layer_call_and_return_conditional_losses_2001?
$batchnorm_31/StatefulPartitionedCallStatefulPartitionedCall2conv_transpose_30/StatefulPartitionedCall:output:0batchnorm_31_3183batchnorm_31_3185batchnorm_31_3187batchnorm_31_3189*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_batchnorm_31_layer_call_and_return_conditional_losses_2030?
leaky_relu_32/PartitionedCallPartitionedCall-batchnorm_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_leaky_relu_32_layer_call_and_return_conditional_losses_2421?
(conv_transpose_a/StatefulPartitionedCallStatefulPartitionedCall&leaky_relu_32/PartitionedCall:output:0conv_transpose_a_3193conv_transpose_a_3195*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_a_layer_call_and_return_conditional_losses_2109?
#batchnorm_b/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_a/StatefulPartitionedCall:output:0batchnorm_b_3198batchnorm_b_3200batchnorm_b_3202batchnorm_b_3204*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_b_layer_call_and_return_conditional_losses_2138?
leaky_relu_c/PartitionedCallPartitionedCall,batchnorm_b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_c_layer_call_and_return_conditional_losses_2442?
)conv_transpose_4a/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_c/PartitionedCall:output:0conv_transpose_4a_3208conv_transpose_4a_3210*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv_transpose_4a_layer_call_and_return_conditional_losses_2218?
IdentityIdentity2conv_transpose_4a/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_3/StatefulPartitionedCall%^batchnorm_31/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall%^batchnorm_4a/StatefulPartitionedCall$^batchnorm_5/StatefulPartitionedCall$^batchnorm_b/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall ^conv_4a/StatefulPartitionedCall)^conv_transpose_1/StatefulPartitionedCall)^conv_transpose_2/StatefulPartitionedCall*^conv_transpose_30/StatefulPartitionedCall*^conv_transpose_4a/StatefulPartitionedCall)^conv_transpose_a/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_3/StatefulPartitionedCall#batchnorm_3/StatefulPartitionedCall2L
$batchnorm_31/StatefulPartitionedCall$batchnorm_31/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2L
$batchnorm_4a/StatefulPartitionedCall$batchnorm_4a/StatefulPartitionedCall2J
#batchnorm_5/StatefulPartitionedCall#batchnorm_5/StatefulPartitionedCall2J
#batchnorm_b/StatefulPartitionedCall#batchnorm_b/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2B
conv_4a/StatefulPartitionedCallconv_4a/StatefulPartitionedCall2T
(conv_transpose_1/StatefulPartitionedCall(conv_transpose_1/StatefulPartitionedCall2T
(conv_transpose_2/StatefulPartitionedCall(conv_transpose_2/StatefulPartitionedCall2V
)conv_transpose_30/StatefulPartitionedCall)conv_transpose_30/StatefulPartitionedCall2V
)conv_transpose_4a/StatefulPartitionedCall)conv_transpose_4a/StatefulPartitionedCall2T
(conv_transpose_a/StatefulPartitionedCall(conv_transpose_a/StatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameinput_layer
?
?
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_4211

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_b_layer_call_and_return_conditional_losses_2169

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_leaky_relu_32_layer_call_fn_4849

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_leaky_relu_32_layer_call_and_return_conditional_losses_2421h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
? 
?
K__inference_conv_transpose_30_layer_call_and_return_conditional_losses_2001

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_4302

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_1642

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?1
__inference__traced_save_5394
file_prefix,
(savev2_conv_1_kernel_read_readvariableop*
&savev2_conv_1_bias_read_readvariableop0
,savev2_batchnorm_1_gamma_read_readvariableop/
+savev2_batchnorm_1_beta_read_readvariableop6
2savev2_batchnorm_1_moving_mean_read_readvariableop:
6savev2_batchnorm_1_moving_variance_read_readvariableop,
(savev2_conv_2_kernel_read_readvariableop*
&savev2_conv_2_bias_read_readvariableop0
,savev2_batchnorm_2_gamma_read_readvariableop/
+savev2_batchnorm_2_beta_read_readvariableop6
2savev2_batchnorm_2_moving_mean_read_readvariableop:
6savev2_batchnorm_2_moving_variance_read_readvariableop,
(savev2_conv_3_kernel_read_readvariableop*
&savev2_conv_3_bias_read_readvariableop0
,savev2_batchnorm_3_gamma_read_readvariableop/
+savev2_batchnorm_3_beta_read_readvariableop6
2savev2_batchnorm_3_moving_mean_read_readvariableop:
6savev2_batchnorm_3_moving_variance_read_readvariableop-
)savev2_conv_4a_kernel_read_readvariableop+
'savev2_conv_4a_bias_read_readvariableop1
-savev2_batchnorm_4a_gamma_read_readvariableop0
,savev2_batchnorm_4a_beta_read_readvariableop7
3savev2_batchnorm_4a_moving_mean_read_readvariableop;
7savev2_batchnorm_4a_moving_variance_read_readvariableop6
2savev2_conv_transpose_1_kernel_read_readvariableop4
0savev2_conv_transpose_1_bias_read_readvariableop0
,savev2_batchnorm_4_gamma_read_readvariableop/
+savev2_batchnorm_4_beta_read_readvariableop6
2savev2_batchnorm_4_moving_mean_read_readvariableop:
6savev2_batchnorm_4_moving_variance_read_readvariableop6
2savev2_conv_transpose_2_kernel_read_readvariableop4
0savev2_conv_transpose_2_bias_read_readvariableop0
,savev2_batchnorm_5_gamma_read_readvariableop/
+savev2_batchnorm_5_beta_read_readvariableop6
2savev2_batchnorm_5_moving_mean_read_readvariableop:
6savev2_batchnorm_5_moving_variance_read_readvariableop7
3savev2_conv_transpose_30_kernel_read_readvariableop5
1savev2_conv_transpose_30_bias_read_readvariableop1
-savev2_batchnorm_31_gamma_read_readvariableop0
,savev2_batchnorm_31_beta_read_readvariableop7
3savev2_batchnorm_31_moving_mean_read_readvariableop;
7savev2_batchnorm_31_moving_variance_read_readvariableop6
2savev2_conv_transpose_a_kernel_read_readvariableop4
0savev2_conv_transpose_a_bias_read_readvariableop0
,savev2_batchnorm_b_gamma_read_readvariableop/
+savev2_batchnorm_b_beta_read_readvariableop6
2savev2_batchnorm_b_moving_mean_read_readvariableop:
6savev2_batchnorm_b_moving_variance_read_readvariableop7
3savev2_conv_transpose_4a_kernel_read_readvariableop5
1savev2_conv_transpose_4a_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop.
*savev2_conv_1_kernel_m_read_readvariableop,
(savev2_conv_1_bias_m_read_readvariableop2
.savev2_batchnorm_1_gamma_m_read_readvariableop1
-savev2_batchnorm_1_beta_m_read_readvariableop.
*savev2_conv_2_kernel_m_read_readvariableop,
(savev2_conv_2_bias_m_read_readvariableop2
.savev2_batchnorm_2_gamma_m_read_readvariableop1
-savev2_batchnorm_2_beta_m_read_readvariableop.
*savev2_conv_3_kernel_m_read_readvariableop,
(savev2_conv_3_bias_m_read_readvariableop2
.savev2_batchnorm_3_gamma_m_read_readvariableop1
-savev2_batchnorm_3_beta_m_read_readvariableop/
+savev2_conv_4a_kernel_m_read_readvariableop-
)savev2_conv_4a_bias_m_read_readvariableop3
/savev2_batchnorm_4a_gamma_m_read_readvariableop2
.savev2_batchnorm_4a_beta_m_read_readvariableop8
4savev2_conv_transpose_1_kernel_m_read_readvariableop6
2savev2_conv_transpose_1_bias_m_read_readvariableop2
.savev2_batchnorm_4_gamma_m_read_readvariableop1
-savev2_batchnorm_4_beta_m_read_readvariableop8
4savev2_conv_transpose_2_kernel_m_read_readvariableop6
2savev2_conv_transpose_2_bias_m_read_readvariableop2
.savev2_batchnorm_5_gamma_m_read_readvariableop1
-savev2_batchnorm_5_beta_m_read_readvariableop9
5savev2_conv_transpose_30_kernel_m_read_readvariableop7
3savev2_conv_transpose_30_bias_m_read_readvariableop3
/savev2_batchnorm_31_gamma_m_read_readvariableop2
.savev2_batchnorm_31_beta_m_read_readvariableop8
4savev2_conv_transpose_a_kernel_m_read_readvariableop6
2savev2_conv_transpose_a_bias_m_read_readvariableop2
.savev2_batchnorm_b_gamma_m_read_readvariableop1
-savev2_batchnorm_b_beta_m_read_readvariableop9
5savev2_conv_transpose_4a_kernel_m_read_readvariableop7
3savev2_conv_transpose_4a_bias_m_read_readvariableop.
*savev2_conv_1_kernel_v_read_readvariableop,
(savev2_conv_1_bias_v_read_readvariableop2
.savev2_batchnorm_1_gamma_v_read_readvariableop1
-savev2_batchnorm_1_beta_v_read_readvariableop.
*savev2_conv_2_kernel_v_read_readvariableop,
(savev2_conv_2_bias_v_read_readvariableop2
.savev2_batchnorm_2_gamma_v_read_readvariableop1
-savev2_batchnorm_2_beta_v_read_readvariableop.
*savev2_conv_3_kernel_v_read_readvariableop,
(savev2_conv_3_bias_v_read_readvariableop2
.savev2_batchnorm_3_gamma_v_read_readvariableop1
-savev2_batchnorm_3_beta_v_read_readvariableop/
+savev2_conv_4a_kernel_v_read_readvariableop-
)savev2_conv_4a_bias_v_read_readvariableop3
/savev2_batchnorm_4a_gamma_v_read_readvariableop2
.savev2_batchnorm_4a_beta_v_read_readvariableop8
4savev2_conv_transpose_1_kernel_v_read_readvariableop6
2savev2_conv_transpose_1_bias_v_read_readvariableop2
.savev2_batchnorm_4_gamma_v_read_readvariableop1
-savev2_batchnorm_4_beta_v_read_readvariableop8
4savev2_conv_transpose_2_kernel_v_read_readvariableop6
2savev2_conv_transpose_2_bias_v_read_readvariableop2
.savev2_batchnorm_5_gamma_v_read_readvariableop1
-savev2_batchnorm_5_beta_v_read_readvariableop9
5savev2_conv_transpose_30_kernel_v_read_readvariableop7
3savev2_conv_transpose_30_bias_v_read_readvariableop3
/savev2_batchnorm_31_gamma_v_read_readvariableop2
.savev2_batchnorm_31_beta_v_read_readvariableop8
4savev2_conv_transpose_a_kernel_v_read_readvariableop6
2savev2_conv_transpose_a_bias_v_read_readvariableop2
.savev2_batchnorm_b_gamma_v_read_readvariableop1
-savev2_batchnorm_b_beta_v_read_readvariableop9
5savev2_conv_transpose_4a_kernel_v_read_readvariableop7
3savev2_conv_transpose_4a_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?D
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:y*
dtype0*?D
value?CB?CyB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:y*
dtype0*?
value?B?yB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?/
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop,savev2_batchnorm_1_gamma_read_readvariableop+savev2_batchnorm_1_beta_read_readvariableop2savev2_batchnorm_1_moving_mean_read_readvariableop6savev2_batchnorm_1_moving_variance_read_readvariableop(savev2_conv_2_kernel_read_readvariableop&savev2_conv_2_bias_read_readvariableop,savev2_batchnorm_2_gamma_read_readvariableop+savev2_batchnorm_2_beta_read_readvariableop2savev2_batchnorm_2_moving_mean_read_readvariableop6savev2_batchnorm_2_moving_variance_read_readvariableop(savev2_conv_3_kernel_read_readvariableop&savev2_conv_3_bias_read_readvariableop,savev2_batchnorm_3_gamma_read_readvariableop+savev2_batchnorm_3_beta_read_readvariableop2savev2_batchnorm_3_moving_mean_read_readvariableop6savev2_batchnorm_3_moving_variance_read_readvariableop)savev2_conv_4a_kernel_read_readvariableop'savev2_conv_4a_bias_read_readvariableop-savev2_batchnorm_4a_gamma_read_readvariableop,savev2_batchnorm_4a_beta_read_readvariableop3savev2_batchnorm_4a_moving_mean_read_readvariableop7savev2_batchnorm_4a_moving_variance_read_readvariableop2savev2_conv_transpose_1_kernel_read_readvariableop0savev2_conv_transpose_1_bias_read_readvariableop,savev2_batchnorm_4_gamma_read_readvariableop+savev2_batchnorm_4_beta_read_readvariableop2savev2_batchnorm_4_moving_mean_read_readvariableop6savev2_batchnorm_4_moving_variance_read_readvariableop2savev2_conv_transpose_2_kernel_read_readvariableop0savev2_conv_transpose_2_bias_read_readvariableop,savev2_batchnorm_5_gamma_read_readvariableop+savev2_batchnorm_5_beta_read_readvariableop2savev2_batchnorm_5_moving_mean_read_readvariableop6savev2_batchnorm_5_moving_variance_read_readvariableop3savev2_conv_transpose_30_kernel_read_readvariableop1savev2_conv_transpose_30_bias_read_readvariableop-savev2_batchnorm_31_gamma_read_readvariableop,savev2_batchnorm_31_beta_read_readvariableop3savev2_batchnorm_31_moving_mean_read_readvariableop7savev2_batchnorm_31_moving_variance_read_readvariableop2savev2_conv_transpose_a_kernel_read_readvariableop0savev2_conv_transpose_a_bias_read_readvariableop,savev2_batchnorm_b_gamma_read_readvariableop+savev2_batchnorm_b_beta_read_readvariableop2savev2_batchnorm_b_moving_mean_read_readvariableop6savev2_batchnorm_b_moving_variance_read_readvariableop3savev2_conv_transpose_4a_kernel_read_readvariableop1savev2_conv_transpose_4a_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop*savev2_conv_1_kernel_m_read_readvariableop(savev2_conv_1_bias_m_read_readvariableop.savev2_batchnorm_1_gamma_m_read_readvariableop-savev2_batchnorm_1_beta_m_read_readvariableop*savev2_conv_2_kernel_m_read_readvariableop(savev2_conv_2_bias_m_read_readvariableop.savev2_batchnorm_2_gamma_m_read_readvariableop-savev2_batchnorm_2_beta_m_read_readvariableop*savev2_conv_3_kernel_m_read_readvariableop(savev2_conv_3_bias_m_read_readvariableop.savev2_batchnorm_3_gamma_m_read_readvariableop-savev2_batchnorm_3_beta_m_read_readvariableop+savev2_conv_4a_kernel_m_read_readvariableop)savev2_conv_4a_bias_m_read_readvariableop/savev2_batchnorm_4a_gamma_m_read_readvariableop.savev2_batchnorm_4a_beta_m_read_readvariableop4savev2_conv_transpose_1_kernel_m_read_readvariableop2savev2_conv_transpose_1_bias_m_read_readvariableop.savev2_batchnorm_4_gamma_m_read_readvariableop-savev2_batchnorm_4_beta_m_read_readvariableop4savev2_conv_transpose_2_kernel_m_read_readvariableop2savev2_conv_transpose_2_bias_m_read_readvariableop.savev2_batchnorm_5_gamma_m_read_readvariableop-savev2_batchnorm_5_beta_m_read_readvariableop5savev2_conv_transpose_30_kernel_m_read_readvariableop3savev2_conv_transpose_30_bias_m_read_readvariableop/savev2_batchnorm_31_gamma_m_read_readvariableop.savev2_batchnorm_31_beta_m_read_readvariableop4savev2_conv_transpose_a_kernel_m_read_readvariableop2savev2_conv_transpose_a_bias_m_read_readvariableop.savev2_batchnorm_b_gamma_m_read_readvariableop-savev2_batchnorm_b_beta_m_read_readvariableop5savev2_conv_transpose_4a_kernel_m_read_readvariableop3savev2_conv_transpose_4a_bias_m_read_readvariableop*savev2_conv_1_kernel_v_read_readvariableop(savev2_conv_1_bias_v_read_readvariableop.savev2_batchnorm_1_gamma_v_read_readvariableop-savev2_batchnorm_1_beta_v_read_readvariableop*savev2_conv_2_kernel_v_read_readvariableop(savev2_conv_2_bias_v_read_readvariableop.savev2_batchnorm_2_gamma_v_read_readvariableop-savev2_batchnorm_2_beta_v_read_readvariableop*savev2_conv_3_kernel_v_read_readvariableop(savev2_conv_3_bias_v_read_readvariableop.savev2_batchnorm_3_gamma_v_read_readvariableop-savev2_batchnorm_3_beta_v_read_readvariableop+savev2_conv_4a_kernel_v_read_readvariableop)savev2_conv_4a_bias_v_read_readvariableop/savev2_batchnorm_4a_gamma_v_read_readvariableop.savev2_batchnorm_4a_beta_v_read_readvariableop4savev2_conv_transpose_1_kernel_v_read_readvariableop2savev2_conv_transpose_1_bias_v_read_readvariableop.savev2_batchnorm_4_gamma_v_read_readvariableop-savev2_batchnorm_4_beta_v_read_readvariableop4savev2_conv_transpose_2_kernel_v_read_readvariableop2savev2_conv_transpose_2_bias_v_read_readvariableop.savev2_batchnorm_5_gamma_v_read_readvariableop-savev2_batchnorm_5_beta_v_read_readvariableop5savev2_conv_transpose_30_kernel_v_read_readvariableop3savev2_conv_transpose_30_bias_v_read_readvariableop/savev2_batchnorm_31_gamma_v_read_readvariableop.savev2_batchnorm_31_beta_v_read_readvariableop4savev2_conv_transpose_a_kernel_v_read_readvariableop2savev2_conv_transpose_a_bias_v_read_readvariableop.savev2_batchnorm_b_gamma_v_read_readvariableop-savev2_batchnorm_b_beta_v_read_readvariableop5savev2_conv_transpose_4a_kernel_v_read_readvariableop3savev2_conv_transpose_4a_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes}
{2y?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::: : ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
::,%(
&
_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
::,+(
&
_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
::,1(
&
_output_shapes
:: 2

_output_shapes
::3

_output_shapes
: :4

_output_shapes
: :,5(
&
_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
::,9(
&
_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
:: <

_output_shapes
::,=(
&
_output_shapes
:: >

_output_shapes
:: ?

_output_shapes
:: @

_output_shapes
::,A(
&
_output_shapes
:: B

_output_shapes
:: C

_output_shapes
:: D

_output_shapes
::,E(
&
_output_shapes
:: F

_output_shapes
:: G

_output_shapes
:: H

_output_shapes
::,I(
&
_output_shapes
:: J

_output_shapes
:: K

_output_shapes
:: L

_output_shapes
::,M(
&
_output_shapes
:: N

_output_shapes
:: O

_output_shapes
:: P

_output_shapes
::,Q(
&
_output_shapes
:: R

_output_shapes
:: S

_output_shapes
:: T

_output_shapes
::,U(
&
_output_shapes
:: V

_output_shapes
::,W(
&
_output_shapes
:: X

_output_shapes
:: Y

_output_shapes
:: Z

_output_shapes
::,[(
&
_output_shapes
:: \

_output_shapes
:: ]

_output_shapes
:: ^

_output_shapes
::,_(
&
_output_shapes
:: `

_output_shapes
:: a

_output_shapes
:: b

_output_shapes
::,c(
&
_output_shapes
:: d

_output_shapes
:: e

_output_shapes
:: f

_output_shapes
::,g(
&
_output_shapes
:: h

_output_shapes
:: i

_output_shapes
:: j

_output_shapes
::,k(
&
_output_shapes
:: l

_output_shapes
:: m

_output_shapes
:: n

_output_shapes
::,o(
&
_output_shapes
:: p

_output_shapes
:: q

_output_shapes
:: r

_output_shapes
::,s(
&
_output_shapes
:: t

_output_shapes
:: u

_output_shapes
:: v

_output_shapes
::,w(
&
_output_shapes
:: x

_output_shapes
::y

_output_shapes
: 
?
?
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_4730

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_1814

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_4393

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_leaky_relu_1_layer_call_and_return_conditional_losses_2262

inputs
identitya
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:???????????*
alpha%???>i
IdentityIdentityLeakyRelu:activations:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_2_layer_call_fn_4271

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_1578?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_b_layer_call_fn_4909

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_b_layer_call_and_return_conditional_losses_2138?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
? 
?
J__inference_conv_transpose_2_layer_call_and_return_conditional_losses_4668

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
? 
?
J__inference_conv_transpose_2_layer_call_and_return_conditional_losses_1893

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
M
input_layer>
serving_default_input_layer:0???????????O
conv_transpose_4a:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer_with_weights-12
layer-19
layer_with_weights-13
layer-20
layer-21
layer_with_weights-14
layer-22
layer_with_weights-15
layer-23
layer-24
layer_with_weights-16
layer-25
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_default_save_signature
#
signatures"
_tf_keras_network
"
_tf_keras_input_layer
?

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
?
,axis
	-gamma
.beta
/moving_mean
0moving_variance
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
?
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
?

=kernel
>bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
?
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Vkernel
Wbias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
?
^axis
	_gamma
`beta
amoving_mean
bmoving_variance
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layer
?
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
?

okernel
pbias
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_layer
?
waxis
	xgamma
ybeta
zmoving_mean
{moving_variance
|	variables
}trainable_variables
~regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?$m?%m?-m?.m?=m?>m?Fm?Gm?Vm?Wm?_m?`m?om?pm?xm?ym?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?$v?%v?-v?.v?=v?>v?Fv?Gv?Vv?Wv?_v?`v?ov?pv?xv?yv?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
?
$0
%1
-2
.3
/4
05
=6
>7
F8
G9
H10
I11
V12
W13
_14
`15
a16
b17
o18
p19
x20
y21
z22
{23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49"
trackable_list_wrapper
?
$0
%1
-2
.3
=4
>5
F6
G7
V8
W9
_10
`11
o12
p13
x14
y15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
"_default_save_signature
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
?2?
$__inference_model_layer_call_fn_2553
$__inference_model_layer_call_fn_3448
$__inference_model_layer_call_fn_3553
$__inference_model_layer_call_fn_3085?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
?__inference_model_layer_call_and_return_conditional_losses_3797
?__inference_model_layer_call_and_return_conditional_losses_4041
?__inference_model_layer_call_and_return_conditional_losses_3214
?__inference_model_layer_call_and_return_conditional_losses_3343?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
__inference__wrapped_model_1492input_layer"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
-
?serving_default"
signature_map
':%2conv_1/kernel
:2conv_1/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_conv_1_layer_call_fn_4157?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_conv_1_layer_call_and_return_conditional_losses_4167?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
:2batchnorm_1/gamma
:2batchnorm_1/beta
':% (2batchnorm_1/moving_mean
+:) (2batchnorm_1/moving_variance
<
-0
.1
/2
03"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_batchnorm_1_layer_call_fn_4180
*__inference_batchnorm_1_layer_call_fn_4193?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_4211
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_4229?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_leaky_relu_1_layer_call_fn_4234?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_leaky_relu_1_layer_call_and_return_conditional_losses_4239?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
':%2conv_2/kernel
:2conv_2/bias
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_conv_2_layer_call_fn_4248?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_conv_2_layer_call_and_return_conditional_losses_4258?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
:2batchnorm_2/gamma
:2batchnorm_2/beta
':% (2batchnorm_2/moving_mean
+:) (2batchnorm_2/moving_variance
<
F0
G1
H2
I3"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_batchnorm_2_layer_call_fn_4271
*__inference_batchnorm_2_layer_call_fn_4284?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_4302
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_4320?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_leaky_relu_2_layer_call_fn_4325?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_leaky_relu_2_layer_call_and_return_conditional_losses_4330?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
':%2conv_3/kernel
:2conv_3/bias
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_conv_3_layer_call_fn_4339?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_conv_3_layer_call_and_return_conditional_losses_4349?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
:2batchnorm_3/gamma
:2batchnorm_3/beta
':% (2batchnorm_3/moving_mean
+:) (2batchnorm_3/moving_variance
<
_0
`1
a2
b3"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_batchnorm_3_layer_call_fn_4362
*__inference_batchnorm_3_layer_call_fn_4375?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_4393
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_4411?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_leaky_relu_3_layer_call_fn_4416?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_leaky_relu_3_layer_call_and_return_conditional_losses_4421?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
(:&2conv_4a/kernel
:2conv_4a/bias
.
o0
p1"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_conv_4a_layer_call_fn_4430?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv_4a_layer_call_and_return_conditional_losses_4440?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 :2batchnorm_4a/gamma
:2batchnorm_4a/beta
(:& (2batchnorm_4a/moving_mean
,:* (2batchnorm_4a/moving_variance
<
x0
y1
z2
{3"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
|	variables
}trainable_variables
~regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_batchnorm_4a_layer_call_fn_4453
+__inference_batchnorm_4a_layer_call_fn_4466?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_batchnorm_4a_layer_call_and_return_conditional_losses_4484
F__inference_batchnorm_4a_layer_call_and_return_conditional_losses_4502?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_leaky_relu_4a_layer_call_fn_4507?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_leaky_relu_4a_layer_call_and_return_conditional_losses_4512?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
1:/2conv_transpose_1/kernel
#:!2conv_transpose_1/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_conv_transpose_1_layer_call_fn_4521?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_conv_transpose_1_layer_call_and_return_conditional_losses_4554?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
:2batchnorm_4/gamma
:2batchnorm_4/beta
':% (2batchnorm_4/moving_mean
+:) (2batchnorm_4/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_batchnorm_4_layer_call_fn_4567
*__inference_batchnorm_4_layer_call_fn_4580?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_4598
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_4616?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_leaky_relu_4_layer_call_fn_4621?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_leaky_relu_4_layer_call_and_return_conditional_losses_4626?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
1:/2conv_transpose_2/kernel
#:!2conv_transpose_2/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_conv_transpose_2_layer_call_fn_4635?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_conv_transpose_2_layer_call_and_return_conditional_losses_4668?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
:2batchnorm_5/gamma
:2batchnorm_5/beta
':% (2batchnorm_5/moving_mean
+:) (2batchnorm_5/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_batchnorm_5_layer_call_fn_4681
*__inference_batchnorm_5_layer_call_fn_4694?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_4712
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_4730?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_leaky_relu_5_layer_call_fn_4735?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_leaky_relu_5_layer_call_and_return_conditional_losses_4740?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
2:02conv_transpose_30/kernel
$:"2conv_transpose_30/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
0__inference_conv_transpose_30_layer_call_fn_4749?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_conv_transpose_30_layer_call_and_return_conditional_losses_4782?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 :2batchnorm_31/gamma
:2batchnorm_31/beta
(:& (2batchnorm_31/moving_mean
,:* (2batchnorm_31/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_batchnorm_31_layer_call_fn_4795
+__inference_batchnorm_31_layer_call_fn_4808?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_batchnorm_31_layer_call_and_return_conditional_losses_4826
F__inference_batchnorm_31_layer_call_and_return_conditional_losses_4844?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_leaky_relu_32_layer_call_fn_4849?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_leaky_relu_32_layer_call_and_return_conditional_losses_4854?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
1:/2conv_transpose_a/kernel
#:!2conv_transpose_a/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_conv_transpose_a_layer_call_fn_4863?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_conv_transpose_a_layer_call_and_return_conditional_losses_4896?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
:2batchnorm_b/gamma
:2batchnorm_b/beta
':% (2batchnorm_b/moving_mean
+:) (2batchnorm_b/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_batchnorm_b_layer_call_fn_4909
*__inference_batchnorm_b_layer_call_fn_4922?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_batchnorm_b_layer_call_and_return_conditional_losses_4940
E__inference_batchnorm_b_layer_call_and_return_conditional_losses_4958?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_leaky_relu_c_layer_call_fn_4963?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_leaky_relu_c_layer_call_and_return_conditional_losses_4968?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
2:02conv_transpose_4a/kernel
$:"2conv_transpose_4a/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
0__inference_conv_transpose_4a_layer_call_fn_4977?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_conv_transpose_4a_layer_call_and_return_conditional_losses_5011?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
/0
01
H2
I3
a4
b5
z6
{7
?8
?9
?10
?11
?12
?13
?14
?15"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
"__inference_signature_wrapper_4148input_layer"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
':%2conv_1/kernel/m
:2conv_1/bias/m
:2batchnorm_1/gamma/m
:2batchnorm_1/beta/m
':%2conv_2/kernel/m
:2conv_2/bias/m
:2batchnorm_2/gamma/m
:2batchnorm_2/beta/m
':%2conv_3/kernel/m
:2conv_3/bias/m
:2batchnorm_3/gamma/m
:2batchnorm_3/beta/m
(:&2conv_4a/kernel/m
:2conv_4a/bias/m
 :2batchnorm_4a/gamma/m
:2batchnorm_4a/beta/m
1:/2conv_transpose_1/kernel/m
#:!2conv_transpose_1/bias/m
:2batchnorm_4/gamma/m
:2batchnorm_4/beta/m
1:/2conv_transpose_2/kernel/m
#:!2conv_transpose_2/bias/m
:2batchnorm_5/gamma/m
:2batchnorm_5/beta/m
2:02conv_transpose_30/kernel/m
$:"2conv_transpose_30/bias/m
 :2batchnorm_31/gamma/m
:2batchnorm_31/beta/m
1:/2conv_transpose_a/kernel/m
#:!2conv_transpose_a/bias/m
:2batchnorm_b/gamma/m
:2batchnorm_b/beta/m
2:02conv_transpose_4a/kernel/m
$:"2conv_transpose_4a/bias/m
':%2conv_1/kernel/v
:2conv_1/bias/v
:2batchnorm_1/gamma/v
:2batchnorm_1/beta/v
':%2conv_2/kernel/v
:2conv_2/bias/v
:2batchnorm_2/gamma/v
:2batchnorm_2/beta/v
':%2conv_3/kernel/v
:2conv_3/bias/v
:2batchnorm_3/gamma/v
:2batchnorm_3/beta/v
(:&2conv_4a/kernel/v
:2conv_4a/bias/v
 :2batchnorm_4a/gamma/v
:2batchnorm_4a/beta/v
1:/2conv_transpose_1/kernel/v
#:!2conv_transpose_1/bias/v
:2batchnorm_4/gamma/v
:2batchnorm_4/beta/v
1:/2conv_transpose_2/kernel/v
#:!2conv_transpose_2/bias/v
:2batchnorm_5/gamma/v
:2batchnorm_5/beta/v
2:02conv_transpose_30/kernel/v
$:"2conv_transpose_30/bias/v
 :2batchnorm_31/gamma/v
:2batchnorm_31/beta/v
1:/2conv_transpose_a/kernel/v
#:!2conv_transpose_a/bias/v
:2batchnorm_b/gamma/v
:2batchnorm_b/beta/v
2:02conv_transpose_4a/kernel/v
$:"2conv_transpose_4a/bias/v?
__inference__wrapped_model_1492?L$%-./0=>FGHIVW_`abopxyz{??????????????????????????>?;
4?1
/?,
input_layer???????????
? "O?L
J
conv_transpose_4a5?2
conv_transpose_4a????????????
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_4211?-./0M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_4229?-./0M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
*__inference_batchnorm_1_layer_call_fn_4180?-./0M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
*__inference_batchnorm_1_layer_call_fn_4193?-./0M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_4302?FGHIM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_4320?FGHIM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
*__inference_batchnorm_2_layer_call_fn_4271?FGHIM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
*__inference_batchnorm_2_layer_call_fn_4284?FGHIM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
F__inference_batchnorm_31_layer_call_and_return_conditional_losses_4826?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
F__inference_batchnorm_31_layer_call_and_return_conditional_losses_4844?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
+__inference_batchnorm_31_layer_call_fn_4795?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
+__inference_batchnorm_31_layer_call_fn_4808?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_4393?_`abM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_4411?_`abM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
*__inference_batchnorm_3_layer_call_fn_4362?_`abM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
*__inference_batchnorm_3_layer_call_fn_4375?_`abM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_4598?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_4616?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
*__inference_batchnorm_4_layer_call_fn_4567?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
*__inference_batchnorm_4_layer_call_fn_4580?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
F__inference_batchnorm_4a_layer_call_and_return_conditional_losses_4484?xyz{M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
F__inference_batchnorm_4a_layer_call_and_return_conditional_losses_4502?xyz{M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
+__inference_batchnorm_4a_layer_call_fn_4453?xyz{M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
+__inference_batchnorm_4a_layer_call_fn_4466?xyz{M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_4712?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_4730?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
*__inference_batchnorm_5_layer_call_fn_4681?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
*__inference_batchnorm_5_layer_call_fn_4694?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
E__inference_batchnorm_b_layer_call_and_return_conditional_losses_4940?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
E__inference_batchnorm_b_layer_call_and_return_conditional_losses_4958?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
*__inference_batchnorm_b_layer_call_fn_4909?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
*__inference_batchnorm_b_layer_call_fn_4922?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
@__inference_conv_1_layer_call_and_return_conditional_losses_4167p$%9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
%__inference_conv_1_layer_call_fn_4157c$%9?6
/?,
*?'
inputs???????????
? ""?????????????
@__inference_conv_2_layer_call_and_return_conditional_losses_4258n=>9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????  
? ?
%__inference_conv_2_layer_call_fn_4248a=>9?6
/?,
*?'
inputs???????????
? " ??????????  ?
@__inference_conv_3_layer_call_and_return_conditional_losses_4349lVW7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????
? ?
%__inference_conv_3_layer_call_fn_4339_VW7?4
-?*
(?%
inputs?????????  
? " ???????????
A__inference_conv_4a_layer_call_and_return_conditional_losses_4440lop7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
&__inference_conv_4a_layer_call_fn_4430_op7?4
-?*
(?%
inputs?????????
? " ???????????
J__inference_conv_transpose_1_layer_call_and_return_conditional_losses_4554???I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
/__inference_conv_transpose_1_layer_call_fn_4521???I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
J__inference_conv_transpose_2_layer_call_and_return_conditional_losses_4668???I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
/__inference_conv_transpose_2_layer_call_fn_4635???I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
K__inference_conv_transpose_30_layer_call_and_return_conditional_losses_4782???I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
0__inference_conv_transpose_30_layer_call_fn_4749???I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
K__inference_conv_transpose_4a_layer_call_and_return_conditional_losses_5011???I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
0__inference_conv_transpose_4a_layer_call_fn_4977???I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
J__inference_conv_transpose_a_layer_call_and_return_conditional_losses_4896???I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
/__inference_conv_transpose_a_layer_call_fn_4863???I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
F__inference_leaky_relu_1_layer_call_and_return_conditional_losses_4239l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
+__inference_leaky_relu_1_layer_call_fn_4234_9?6
/?,
*?'
inputs???????????
? ""?????????????
F__inference_leaky_relu_2_layer_call_and_return_conditional_losses_4330h7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
+__inference_leaky_relu_2_layer_call_fn_4325[7?4
-?*
(?%
inputs?????????  
? " ??????????  ?
G__inference_leaky_relu_32_layer_call_and_return_conditional_losses_4854h7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????@@
? ?
,__inference_leaky_relu_32_layer_call_fn_4849[7?4
-?*
(?%
inputs?????????@@
? " ??????????@@?
F__inference_leaky_relu_3_layer_call_and_return_conditional_losses_4421h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
+__inference_leaky_relu_3_layer_call_fn_4416[7?4
-?*
(?%
inputs?????????
? " ???????????
F__inference_leaky_relu_4_layer_call_and_return_conditional_losses_4626h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
+__inference_leaky_relu_4_layer_call_fn_4621[7?4
-?*
(?%
inputs?????????
? " ???????????
G__inference_leaky_relu_4a_layer_call_and_return_conditional_losses_4512h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
,__inference_leaky_relu_4a_layer_call_fn_4507[7?4
-?*
(?%
inputs?????????
? " ???????????
F__inference_leaky_relu_5_layer_call_and_return_conditional_losses_4740h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
+__inference_leaky_relu_5_layer_call_fn_4735[7?4
-?*
(?%
inputs?????????
? " ???????????
F__inference_leaky_relu_c_layer_call_and_return_conditional_losses_4968l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
+__inference_leaky_relu_c_layer_call_fn_4963_9?6
/?,
*?'
inputs???????????
? ""?????????????
?__inference_model_layer_call_and_return_conditional_losses_3214?L$%-./0=>FGHIVW_`abopxyz{??????????????????????????F?C
<?9
/?,
input_layer???????????
p 

 
? "/?,
%?"
0???????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_3343?L$%-./0=>FGHIVW_`abopxyz{??????????????????????????F?C
<?9
/?,
input_layer???????????
p

 
? "/?,
%?"
0???????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_3797?L$%-./0=>FGHIVW_`abopxyz{??????????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_4041?L$%-./0=>FGHIVW_`abopxyz{??????????????????????????A?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????
? ?
$__inference_model_layer_call_fn_2553?L$%-./0=>FGHIVW_`abopxyz{??????????????????????????F?C
<?9
/?,
input_layer???????????
p 

 
? ""?????????????
$__inference_model_layer_call_fn_3085?L$%-./0=>FGHIVW_`abopxyz{??????????????????????????F?C
<?9
/?,
input_layer???????????
p

 
? ""?????????????
$__inference_model_layer_call_fn_3448?L$%-./0=>FGHIVW_`abopxyz{??????????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? ""?????????????
$__inference_model_layer_call_fn_3553?L$%-./0=>FGHIVW_`abopxyz{??????????????????????????A?>
7?4
*?'
inputs???????????
p

 
? ""?????????????
"__inference_signature_wrapper_4148?L$%-./0=>FGHIVW_`abopxyz{??????????????????????????M?J
? 
C?@
>
input_layer/?,
input_layer???????????"O?L
J
conv_transpose_4a5?2
conv_transpose_4a???????????