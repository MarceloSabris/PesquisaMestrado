˼ 
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
executor_typestring ?
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
~
conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv_1/kernel
w
!conv_1/kernel/Read/ReadVariableOpReadVariableOpconv_1/kernel*&
_output_shapes
: *
dtype0
n
conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv_1/bias
g
conv_1/bias/Read/ReadVariableOpReadVariableOpconv_1/bias*
_output_shapes
: *
dtype0
z
batchnorm_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namebatchnorm_1/gamma
s
%batchnorm_1/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_1/gamma*
_output_shapes
: *
dtype0
x
batchnorm_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_namebatchnorm_1/beta
q
$batchnorm_1/beta/Read/ReadVariableOpReadVariableOpbatchnorm_1/beta*
_output_shapes
: *
dtype0
?
batchnorm_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namebatchnorm_1/moving_mean

+batchnorm_1/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_1/moving_mean*
_output_shapes
: *
dtype0
?
batchnorm_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatchnorm_1/moving_variance
?
/batchnorm_1/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_1/moving_variance*
_output_shapes
: *
dtype0
~
conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_nameconv_2/kernel
w
!conv_2/kernel/Read/ReadVariableOpReadVariableOpconv_2/kernel*&
_output_shapes
: @*
dtype0
n
conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_2/bias
g
conv_2/bias/Read/ReadVariableOpReadVariableOpconv_2/bias*
_output_shapes
:@*
dtype0
z
batchnorm_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namebatchnorm_2/gamma
s
%batchnorm_2/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_2/gamma*
_output_shapes
:@*
dtype0
x
batchnorm_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_namebatchnorm_2/beta
q
$batchnorm_2/beta/Read/ReadVariableOpReadVariableOpbatchnorm_2/beta*
_output_shapes
:@*
dtype0
?
batchnorm_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_namebatchnorm_2/moving_mean

+batchnorm_2/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_2/moving_mean*
_output_shapes
:@*
dtype0
?
batchnorm_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatchnorm_2/moving_variance
?
/batchnorm_2/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_2/moving_variance*
_output_shapes
:@*
dtype0
~
conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_nameconv_3/kernel
w
!conv_3/kernel/Read/ReadVariableOpReadVariableOpconv_3/kernel*&
_output_shapes
:@@*
dtype0
n
conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_3/bias
g
conv_3/bias/Read/ReadVariableOpReadVariableOpconv_3/bias*
_output_shapes
:@*
dtype0
z
batchnorm_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namebatchnorm_3/gamma
s
%batchnorm_3/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_3/gamma*
_output_shapes
:@*
dtype0
x
batchnorm_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_namebatchnorm_3/beta
q
$batchnorm_3/beta/Read/ReadVariableOpReadVariableOpbatchnorm_3/beta*
_output_shapes
:@*
dtype0
?
batchnorm_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_namebatchnorm_3/moving_mean

+batchnorm_3/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_3/moving_mean*
_output_shapes
:@*
dtype0
?
batchnorm_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatchnorm_3/moving_variance
?
/batchnorm_3/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_3/moving_variance*
_output_shapes
:@*
dtype0
?
conv_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameconv_transpose_1/kernel
?
+conv_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv_transpose_1/kernel*&
_output_shapes
:@@*
dtype0
?
conv_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameconv_transpose_1/bias
{
)conv_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv_transpose_1/bias*
_output_shapes
:@*
dtype0
z
batchnorm_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namebatchnorm_4/gamma
s
%batchnorm_4/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_4/gamma*
_output_shapes
:@*
dtype0
x
batchnorm_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_namebatchnorm_4/beta
q
$batchnorm_4/beta/Read/ReadVariableOpReadVariableOpbatchnorm_4/beta*
_output_shapes
:@*
dtype0
?
batchnorm_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_namebatchnorm_4/moving_mean

+batchnorm_4/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_4/moving_mean*
_output_shapes
:@*
dtype0
?
batchnorm_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatchnorm_4/moving_variance
?
/batchnorm_4/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_4/moving_variance*
_output_shapes
:@*
dtype0
?
conv_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameconv_transpose_2/kernel
?
+conv_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv_transpose_2/kernel*&
_output_shapes
:@@*
dtype0
?
conv_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameconv_transpose_2/bias
{
)conv_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv_transpose_2/bias*
_output_shapes
:@*
dtype0
z
batchnorm_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namebatchnorm_5/gamma
s
%batchnorm_5/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_5/gamma*
_output_shapes
:@*
dtype0
x
batchnorm_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_namebatchnorm_5/beta
q
$batchnorm_5/beta/Read/ReadVariableOpReadVariableOpbatchnorm_5/beta*
_output_shapes
:@*
dtype0
?
batchnorm_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_namebatchnorm_5/moving_mean

+batchnorm_5/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_5/moving_mean*
_output_shapes
:@*
dtype0
?
batchnorm_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatchnorm_5/moving_variance
?
/batchnorm_5/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_5/moving_variance*
_output_shapes
:@*
dtype0
?
conv_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameconv_transpose_3/kernel
?
+conv_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv_transpose_3/kernel*&
_output_shapes
: @*
dtype0
?
conv_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameconv_transpose_3/bias
{
)conv_transpose_3/bias/Read/ReadVariableOpReadVariableOpconv_transpose_3/bias*
_output_shapes
: *
dtype0
z
batchnorm_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namebatchnorm_6/gamma
s
%batchnorm_6/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_6/gamma*
_output_shapes
: *
dtype0
x
batchnorm_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_namebatchnorm_6/beta
q
$batchnorm_6/beta/Read/ReadVariableOpReadVariableOpbatchnorm_6/beta*
_output_shapes
: *
dtype0
?
batchnorm_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namebatchnorm_6/moving_mean

+batchnorm_6/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_6/moving_mean*
_output_shapes
: *
dtype0
?
batchnorm_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatchnorm_6/moving_variance
?
/batchnorm_6/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_6/moving_variance*
_output_shapes
: *
dtype0
?
conv_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv_transpose_4/kernel
?
+conv_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv_transpose_4/kernel*&
_output_shapes
: *
dtype0
?
conv_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv_transpose_4/bias
{
)conv_transpose_4/bias/Read/ReadVariableOpReadVariableOpconv_transpose_4/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
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
Adam/conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv_1/kernel/m
?
(Adam/conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_1/kernel/m*&
_output_shapes
: *
dtype0
|
Adam/conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv_1/bias/m
u
&Adam/conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_1/bias/m*
_output_shapes
: *
dtype0
?
Adam/batchnorm_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/batchnorm_1/gamma/m
?
,Adam/batchnorm_1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_1/gamma/m*
_output_shapes
: *
dtype0
?
Adam/batchnorm_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/batchnorm_1/beta/m

+Adam/batchnorm_1/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_1/beta/m*
_output_shapes
: *
dtype0
?
Adam/conv_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*%
shared_nameAdam/conv_2/kernel/m
?
(Adam/conv_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_2/kernel/m*&
_output_shapes
: @*
dtype0
|
Adam/conv_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv_2/bias/m
u
&Adam/conv_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_2/bias/m*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/batchnorm_2/gamma/m
?
,Adam/batchnorm_2/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_2/gamma/m*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/batchnorm_2/beta/m

+Adam/batchnorm_2/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_2/beta/m*
_output_shapes
:@*
dtype0
?
Adam/conv_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*%
shared_nameAdam/conv_3/kernel/m
?
(Adam/conv_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_3/kernel/m*&
_output_shapes
:@@*
dtype0
|
Adam/conv_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv_3/bias/m
u
&Adam/conv_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_3/bias/m*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/batchnorm_3/gamma/m
?
,Adam/batchnorm_3/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_3/gamma/m*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/batchnorm_3/beta/m

+Adam/batchnorm_3/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_3/beta/m*
_output_shapes
:@*
dtype0
?
Adam/conv_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*/
shared_name Adam/conv_transpose_1/kernel/m
?
2Adam/conv_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_transpose_1/kernel/m*&
_output_shapes
:@@*
dtype0
?
Adam/conv_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/conv_transpose_1/bias/m
?
0Adam/conv_transpose_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_transpose_1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/batchnorm_4/gamma/m
?
,Adam/batchnorm_4/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_4/gamma/m*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/batchnorm_4/beta/m

+Adam/batchnorm_4/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_4/beta/m*
_output_shapes
:@*
dtype0
?
Adam/conv_transpose_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*/
shared_name Adam/conv_transpose_2/kernel/m
?
2Adam/conv_transpose_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_transpose_2/kernel/m*&
_output_shapes
:@@*
dtype0
?
Adam/conv_transpose_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/conv_transpose_2/bias/m
?
0Adam/conv_transpose_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_transpose_2/bias/m*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/batchnorm_5/gamma/m
?
,Adam/batchnorm_5/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_5/gamma/m*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/batchnorm_5/beta/m

+Adam/batchnorm_5/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_5/beta/m*
_output_shapes
:@*
dtype0
?
Adam/conv_transpose_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*/
shared_name Adam/conv_transpose_3/kernel/m
?
2Adam/conv_transpose_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_transpose_3/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv_transpose_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/conv_transpose_3/bias/m
?
0Adam/conv_transpose_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_transpose_3/bias/m*
_output_shapes
: *
dtype0
?
Adam/batchnorm_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/batchnorm_6/gamma/m
?
,Adam/batchnorm_6/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_6/gamma/m*
_output_shapes
: *
dtype0
?
Adam/batchnorm_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/batchnorm_6/beta/m

+Adam/batchnorm_6/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_6/beta/m*
_output_shapes
: *
dtype0
?
Adam/conv_transpose_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/conv_transpose_4/kernel/m
?
2Adam/conv_transpose_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_transpose_4/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv_transpose_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/conv_transpose_4/bias/m
?
0Adam/conv_transpose_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_transpose_4/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv_1/kernel/v
?
(Adam/conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_1/kernel/v*&
_output_shapes
: *
dtype0
|
Adam/conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv_1/bias/v
u
&Adam/conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_1/bias/v*
_output_shapes
: *
dtype0
?
Adam/batchnorm_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/batchnorm_1/gamma/v
?
,Adam/batchnorm_1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_1/gamma/v*
_output_shapes
: *
dtype0
?
Adam/batchnorm_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/batchnorm_1/beta/v

+Adam/batchnorm_1/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_1/beta/v*
_output_shapes
: *
dtype0
?
Adam/conv_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*%
shared_nameAdam/conv_2/kernel/v
?
(Adam/conv_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_2/kernel/v*&
_output_shapes
: @*
dtype0
|
Adam/conv_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv_2/bias/v
u
&Adam/conv_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_2/bias/v*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/batchnorm_2/gamma/v
?
,Adam/batchnorm_2/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_2/gamma/v*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/batchnorm_2/beta/v

+Adam/batchnorm_2/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_2/beta/v*
_output_shapes
:@*
dtype0
?
Adam/conv_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*%
shared_nameAdam/conv_3/kernel/v
?
(Adam/conv_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_3/kernel/v*&
_output_shapes
:@@*
dtype0
|
Adam/conv_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv_3/bias/v
u
&Adam/conv_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_3/bias/v*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/batchnorm_3/gamma/v
?
,Adam/batchnorm_3/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_3/gamma/v*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/batchnorm_3/beta/v

+Adam/batchnorm_3/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_3/beta/v*
_output_shapes
:@*
dtype0
?
Adam/conv_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*/
shared_name Adam/conv_transpose_1/kernel/v
?
2Adam/conv_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_transpose_1/kernel/v*&
_output_shapes
:@@*
dtype0
?
Adam/conv_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/conv_transpose_1/bias/v
?
0Adam/conv_transpose_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_transpose_1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/batchnorm_4/gamma/v
?
,Adam/batchnorm_4/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_4/gamma/v*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/batchnorm_4/beta/v

+Adam/batchnorm_4/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_4/beta/v*
_output_shapes
:@*
dtype0
?
Adam/conv_transpose_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*/
shared_name Adam/conv_transpose_2/kernel/v
?
2Adam/conv_transpose_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_transpose_2/kernel/v*&
_output_shapes
:@@*
dtype0
?
Adam/conv_transpose_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/conv_transpose_2/bias/v
?
0Adam/conv_transpose_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_transpose_2/bias/v*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/batchnorm_5/gamma/v
?
,Adam/batchnorm_5/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_5/gamma/v*
_output_shapes
:@*
dtype0
?
Adam/batchnorm_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/batchnorm_5/beta/v

+Adam/batchnorm_5/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_5/beta/v*
_output_shapes
:@*
dtype0
?
Adam/conv_transpose_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*/
shared_name Adam/conv_transpose_3/kernel/v
?
2Adam/conv_transpose_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_transpose_3/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv_transpose_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/conv_transpose_3/bias/v
?
0Adam/conv_transpose_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_transpose_3/bias/v*
_output_shapes
: *
dtype0
?
Adam/batchnorm_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/batchnorm_6/gamma/v
?
,Adam/batchnorm_6/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_6/gamma/v*
_output_shapes
: *
dtype0
?
Adam/batchnorm_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/batchnorm_6/beta/v

+Adam/batchnorm_6/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_6/beta/v*
_output_shapes
: *
dtype0
?
Adam/conv_transpose_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/conv_transpose_4/kernel/v
?
2Adam/conv_transpose_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_transpose_4/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv_transpose_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/conv_transpose_4/bias/v
?
0Adam/conv_transpose_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_transpose_4/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
˜
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
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
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
	variables
trainable_variables
 	keras_api
?
!axis
	"gamma
#beta
$moving_mean
%moving_variance
&regularization_losses
'	variables
(trainable_variables
)	keras_api
R
*regularization_losses
+	variables
,trainable_variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
?
4axis
	5gamma
6beta
7moving_mean
8moving_variance
9regularization_losses
:	variables
;trainable_variables
<	keras_api
R
=regularization_losses
>	variables
?trainable_variables
@	keras_api
h

Akernel
Bbias
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
?
Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
R
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
h

Tkernel
Ubias
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
?
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
_regularization_losses
`	variables
atrainable_variables
b	keras_api
R
cregularization_losses
d	variables
etrainable_variables
f	keras_api
h

gkernel
hbias
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
?
maxis
	ngamma
obeta
pmoving_mean
qmoving_variance
rregularization_losses
s	variables
ttrainable_variables
u	keras_api
R
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
h

zkernel
{bias
|regularization_losses
}	variables
~trainable_variables
	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem?m?"m?#m?.m?/m?5m?6m?Am?Bm?Hm?Im?Tm?Um?[m?\m?gm?hm?nm?om?zm?{m?	?m?	?m?	?m?	?m?v?v?"v?#v?.v?/v?5v?6v?Av?Bv?Hv?Iv?Tv?Uv?[v?\v?gv?hv?nv?ov?zv?{v?	?v?	?v?	?v?	?v?
 
?
0
1
"2
#3
$4
%5
.6
/7
58
69
710
811
A12
B13
H14
I15
J16
K17
T18
U19
[20
\21
]22
^23
g24
h25
n26
o27
p28
q29
z30
{31
?32
?33
?34
?35
?36
?37
?
0
1
"2
#3
.4
/5
56
67
A8
B9
H10
I11
T12
U13
[14
\15
g16
h17
n18
o19
z20
{21
?22
?23
?24
?25
?
?non_trainable_variables
regularization_losses
	variables
 ?layer_regularization_losses
?metrics
trainable_variables
?layer_metrics
?layers
 
YW
VARIABLE_VALUEconv_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
?non_trainable_variables
regularization_losses
	variables
 ?layer_regularization_losses
?metrics
trainable_variables
?layer_metrics
?layers
 
\Z
VARIABLE_VALUEbatchnorm_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEbatchnorm_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbatchnorm_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatchnorm_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1
$2
%3

"0
#1
?
?non_trainable_variables
&regularization_losses
'	variables
 ?layer_regularization_losses
?metrics
(trainable_variables
?layer_metrics
?layers
 
 
 
?
?non_trainable_variables
*regularization_losses
+	variables
 ?layer_regularization_losses
?metrics
,trainable_variables
?layer_metrics
?layers
YW
VARIABLE_VALUEconv_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
?
?non_trainable_variables
0regularization_losses
1	variables
 ?layer_regularization_losses
?metrics
2trainable_variables
?layer_metrics
?layers
 
\Z
VARIABLE_VALUEbatchnorm_2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEbatchnorm_2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbatchnorm_2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatchnorm_2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

50
61
72
83

50
61
?
?non_trainable_variables
9regularization_losses
:	variables
 ?layer_regularization_losses
?metrics
;trainable_variables
?layer_metrics
?layers
 
 
 
?
?non_trainable_variables
=regularization_losses
>	variables
 ?layer_regularization_losses
?metrics
?trainable_variables
?layer_metrics
?layers
YW
VARIABLE_VALUEconv_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

A0
B1

A0
B1
?
?non_trainable_variables
Cregularization_losses
D	variables
 ?layer_regularization_losses
?metrics
Etrainable_variables
?layer_metrics
?layers
 
\Z
VARIABLE_VALUEbatchnorm_3/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEbatchnorm_3/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbatchnorm_3/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatchnorm_3/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

H0
I1
J2
K3

H0
I1
?
?non_trainable_variables
Lregularization_losses
M	variables
 ?layer_regularization_losses
?metrics
Ntrainable_variables
?layer_metrics
?layers
 
 
 
?
?non_trainable_variables
Pregularization_losses
Q	variables
 ?layer_regularization_losses
?metrics
Rtrainable_variables
?layer_metrics
?layers
ca
VARIABLE_VALUEconv_transpose_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv_transpose_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

T0
U1

T0
U1
?
?non_trainable_variables
Vregularization_losses
W	variables
 ?layer_regularization_losses
?metrics
Xtrainable_variables
?layer_metrics
?layers
 
\Z
VARIABLE_VALUEbatchnorm_4/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEbatchnorm_4/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbatchnorm_4/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatchnorm_4/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

[0
\1
]2
^3

[0
\1
?
?non_trainable_variables
_regularization_losses
`	variables
 ?layer_regularization_losses
?metrics
atrainable_variables
?layer_metrics
?layers
 
 
 
?
?non_trainable_variables
cregularization_losses
d	variables
 ?layer_regularization_losses
?metrics
etrainable_variables
?layer_metrics
?layers
ca
VARIABLE_VALUEconv_transpose_2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv_transpose_2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

g0
h1

g0
h1
?
?non_trainable_variables
iregularization_losses
j	variables
 ?layer_regularization_losses
?metrics
ktrainable_variables
?layer_metrics
?layers
 
\Z
VARIABLE_VALUEbatchnorm_5/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEbatchnorm_5/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbatchnorm_5/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatchnorm_5/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

n0
o1
p2
q3

n0
o1
?
?non_trainable_variables
rregularization_losses
s	variables
 ?layer_regularization_losses
?metrics
ttrainable_variables
?layer_metrics
?layers
 
 
 
?
?non_trainable_variables
vregularization_losses
w	variables
 ?layer_regularization_losses
?metrics
xtrainable_variables
?layer_metrics
?layers
db
VARIABLE_VALUEconv_transpose_3/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv_transpose_3/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

z0
{1

z0
{1
?
?non_trainable_variables
|regularization_losses
}	variables
 ?layer_regularization_losses
?metrics
~trainable_variables
?layer_metrics
?layers
 
][
VARIABLE_VALUEbatchnorm_6/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatchnorm_6/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEbatchnorm_6/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEbatchnorm_6/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 
?0
?1
?2
?3

?0
?1
?
?non_trainable_variables
?regularization_losses
?	variables
 ?layer_regularization_losses
?metrics
?trainable_variables
?layer_metrics
?layers
 
 
 
?
?non_trainable_variables
?regularization_losses
?	variables
 ?layer_regularization_losses
?metrics
?trainable_variables
?layer_metrics
?layers
db
VARIABLE_VALUEconv_transpose_4/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEconv_transpose_4/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?non_trainable_variables
?regularization_losses
?	variables
 ?layer_regularization_losses
?metrics
?trainable_variables
?layer_metrics
?layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
X
$0
%1
72
83
J4
K5
]6
^7
p8
q9
?10
?11
 

?0
 
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
 
 
 
 
 

$0
%1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

70
81
 
 
 
 
 
 
 
 
 
 
 
 
 
 

J0
K1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

]0
^1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

p0
q1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
|z
VARIABLE_VALUEAdam/conv_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_1/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_1/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_2/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_2/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_3/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_3/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_3/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_3/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_transpose_1/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_transpose_1/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_4/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_4/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_transpose_2/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_transpose_2/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_5/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_5/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_transpose_3/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_transpose_3/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/batchnorm_6/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/batchnorm_6/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_transpose_4/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_transpose_4/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_1/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_1/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_2/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_2/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv_3/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv_3/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_3/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_3/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_transpose_1/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_transpose_1/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_4/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_4/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_transpose_2/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_transpose_2/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_5/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_5/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_transpose_3/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_transpose_3/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/batchnorm_6/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/batchnorm_6/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_transpose_4/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv_transpose_4/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_layerPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerconv_1/kernelconv_1/biasbatchnorm_1/gammabatchnorm_1/betabatchnorm_1/moving_meanbatchnorm_1/moving_varianceconv_2/kernelconv_2/biasbatchnorm_2/gammabatchnorm_2/betabatchnorm_2/moving_meanbatchnorm_2/moving_varianceconv_3/kernelconv_3/biasbatchnorm_3/gammabatchnorm_3/betabatchnorm_3/moving_meanbatchnorm_3/moving_varianceconv_transpose_1/kernelconv_transpose_1/biasbatchnorm_4/gammabatchnorm_4/betabatchnorm_4/moving_meanbatchnorm_4/moving_varianceconv_transpose_2/kernelconv_transpose_2/biasbatchnorm_5/gammabatchnorm_5/betabatchnorm_5/moving_meanbatchnorm_5/moving_varianceconv_transpose_3/kernelconv_transpose_3/biasbatchnorm_6/gammabatchnorm_6/betabatchnorm_6/moving_meanbatchnorm_6/moving_varianceconv_transpose_4/kernelconv_transpose_4/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_6265
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?#
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv_1/kernel/Read/ReadVariableOpconv_1/bias/Read/ReadVariableOp%batchnorm_1/gamma/Read/ReadVariableOp$batchnorm_1/beta/Read/ReadVariableOp+batchnorm_1/moving_mean/Read/ReadVariableOp/batchnorm_1/moving_variance/Read/ReadVariableOp!conv_2/kernel/Read/ReadVariableOpconv_2/bias/Read/ReadVariableOp%batchnorm_2/gamma/Read/ReadVariableOp$batchnorm_2/beta/Read/ReadVariableOp+batchnorm_2/moving_mean/Read/ReadVariableOp/batchnorm_2/moving_variance/Read/ReadVariableOp!conv_3/kernel/Read/ReadVariableOpconv_3/bias/Read/ReadVariableOp%batchnorm_3/gamma/Read/ReadVariableOp$batchnorm_3/beta/Read/ReadVariableOp+batchnorm_3/moving_mean/Read/ReadVariableOp/batchnorm_3/moving_variance/Read/ReadVariableOp+conv_transpose_1/kernel/Read/ReadVariableOp)conv_transpose_1/bias/Read/ReadVariableOp%batchnorm_4/gamma/Read/ReadVariableOp$batchnorm_4/beta/Read/ReadVariableOp+batchnorm_4/moving_mean/Read/ReadVariableOp/batchnorm_4/moving_variance/Read/ReadVariableOp+conv_transpose_2/kernel/Read/ReadVariableOp)conv_transpose_2/bias/Read/ReadVariableOp%batchnorm_5/gamma/Read/ReadVariableOp$batchnorm_5/beta/Read/ReadVariableOp+batchnorm_5/moving_mean/Read/ReadVariableOp/batchnorm_5/moving_variance/Read/ReadVariableOp+conv_transpose_3/kernel/Read/ReadVariableOp)conv_transpose_3/bias/Read/ReadVariableOp%batchnorm_6/gamma/Read/ReadVariableOp$batchnorm_6/beta/Read/ReadVariableOp+batchnorm_6/moving_mean/Read/ReadVariableOp/batchnorm_6/moving_variance/Read/ReadVariableOp+conv_transpose_4/kernel/Read/ReadVariableOp)conv_transpose_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv_1/kernel/m/Read/ReadVariableOp&Adam/conv_1/bias/m/Read/ReadVariableOp,Adam/batchnorm_1/gamma/m/Read/ReadVariableOp+Adam/batchnorm_1/beta/m/Read/ReadVariableOp(Adam/conv_2/kernel/m/Read/ReadVariableOp&Adam/conv_2/bias/m/Read/ReadVariableOp,Adam/batchnorm_2/gamma/m/Read/ReadVariableOp+Adam/batchnorm_2/beta/m/Read/ReadVariableOp(Adam/conv_3/kernel/m/Read/ReadVariableOp&Adam/conv_3/bias/m/Read/ReadVariableOp,Adam/batchnorm_3/gamma/m/Read/ReadVariableOp+Adam/batchnorm_3/beta/m/Read/ReadVariableOp2Adam/conv_transpose_1/kernel/m/Read/ReadVariableOp0Adam/conv_transpose_1/bias/m/Read/ReadVariableOp,Adam/batchnorm_4/gamma/m/Read/ReadVariableOp+Adam/batchnorm_4/beta/m/Read/ReadVariableOp2Adam/conv_transpose_2/kernel/m/Read/ReadVariableOp0Adam/conv_transpose_2/bias/m/Read/ReadVariableOp,Adam/batchnorm_5/gamma/m/Read/ReadVariableOp+Adam/batchnorm_5/beta/m/Read/ReadVariableOp2Adam/conv_transpose_3/kernel/m/Read/ReadVariableOp0Adam/conv_transpose_3/bias/m/Read/ReadVariableOp,Adam/batchnorm_6/gamma/m/Read/ReadVariableOp+Adam/batchnorm_6/beta/m/Read/ReadVariableOp2Adam/conv_transpose_4/kernel/m/Read/ReadVariableOp0Adam/conv_transpose_4/bias/m/Read/ReadVariableOp(Adam/conv_1/kernel/v/Read/ReadVariableOp&Adam/conv_1/bias/v/Read/ReadVariableOp,Adam/batchnorm_1/gamma/v/Read/ReadVariableOp+Adam/batchnorm_1/beta/v/Read/ReadVariableOp(Adam/conv_2/kernel/v/Read/ReadVariableOp&Adam/conv_2/bias/v/Read/ReadVariableOp,Adam/batchnorm_2/gamma/v/Read/ReadVariableOp+Adam/batchnorm_2/beta/v/Read/ReadVariableOp(Adam/conv_3/kernel/v/Read/ReadVariableOp&Adam/conv_3/bias/v/Read/ReadVariableOp,Adam/batchnorm_3/gamma/v/Read/ReadVariableOp+Adam/batchnorm_3/beta/v/Read/ReadVariableOp2Adam/conv_transpose_1/kernel/v/Read/ReadVariableOp0Adam/conv_transpose_1/bias/v/Read/ReadVariableOp,Adam/batchnorm_4/gamma/v/Read/ReadVariableOp+Adam/batchnorm_4/beta/v/Read/ReadVariableOp2Adam/conv_transpose_2/kernel/v/Read/ReadVariableOp0Adam/conv_transpose_2/bias/v/Read/ReadVariableOp,Adam/batchnorm_5/gamma/v/Read/ReadVariableOp+Adam/batchnorm_5/beta/v/Read/ReadVariableOp2Adam/conv_transpose_3/kernel/v/Read/ReadVariableOp0Adam/conv_transpose_3/bias/v/Read/ReadVariableOp,Adam/batchnorm_6/gamma/v/Read/ReadVariableOp+Adam/batchnorm_6/beta/v/Read/ReadVariableOp2Adam/conv_transpose_4/kernel/v/Read/ReadVariableOp0Adam/conv_transpose_4/bias/v/Read/ReadVariableOpConst*n
Ting
e2c	*
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
__inference__traced_save_7824
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_1/kernelconv_1/biasbatchnorm_1/gammabatchnorm_1/betabatchnorm_1/moving_meanbatchnorm_1/moving_varianceconv_2/kernelconv_2/biasbatchnorm_2/gammabatchnorm_2/betabatchnorm_2/moving_meanbatchnorm_2/moving_varianceconv_3/kernelconv_3/biasbatchnorm_3/gammabatchnorm_3/betabatchnorm_3/moving_meanbatchnorm_3/moving_varianceconv_transpose_1/kernelconv_transpose_1/biasbatchnorm_4/gammabatchnorm_4/betabatchnorm_4/moving_meanbatchnorm_4/moving_varianceconv_transpose_2/kernelconv_transpose_2/biasbatchnorm_5/gammabatchnorm_5/betabatchnorm_5/moving_meanbatchnorm_5/moving_varianceconv_transpose_3/kernelconv_transpose_3/biasbatchnorm_6/gammabatchnorm_6/betabatchnorm_6/moving_meanbatchnorm_6/moving_varianceconv_transpose_4/kernelconv_transpose_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv_1/kernel/mAdam/conv_1/bias/mAdam/batchnorm_1/gamma/mAdam/batchnorm_1/beta/mAdam/conv_2/kernel/mAdam/conv_2/bias/mAdam/batchnorm_2/gamma/mAdam/batchnorm_2/beta/mAdam/conv_3/kernel/mAdam/conv_3/bias/mAdam/batchnorm_3/gamma/mAdam/batchnorm_3/beta/mAdam/conv_transpose_1/kernel/mAdam/conv_transpose_1/bias/mAdam/batchnorm_4/gamma/mAdam/batchnorm_4/beta/mAdam/conv_transpose_2/kernel/mAdam/conv_transpose_2/bias/mAdam/batchnorm_5/gamma/mAdam/batchnorm_5/beta/mAdam/conv_transpose_3/kernel/mAdam/conv_transpose_3/bias/mAdam/batchnorm_6/gamma/mAdam/batchnorm_6/beta/mAdam/conv_transpose_4/kernel/mAdam/conv_transpose_4/bias/mAdam/conv_1/kernel/vAdam/conv_1/bias/vAdam/batchnorm_1/gamma/vAdam/batchnorm_1/beta/vAdam/conv_2/kernel/vAdam/conv_2/bias/vAdam/batchnorm_2/gamma/vAdam/batchnorm_2/beta/vAdam/conv_3/kernel/vAdam/conv_3/bias/vAdam/batchnorm_3/gamma/vAdam/batchnorm_3/beta/vAdam/conv_transpose_1/kernel/vAdam/conv_transpose_1/bias/vAdam/batchnorm_4/gamma/vAdam/batchnorm_4/beta/vAdam/conv_transpose_2/kernel/vAdam/conv_transpose_2/bias/vAdam/batchnorm_5/gamma/vAdam/batchnorm_5/beta/vAdam/conv_transpose_3/kernel/vAdam/conv_transpose_3/bias/vAdam/batchnorm_6/gamma/vAdam/batchnorm_6/beta/vAdam/conv_transpose_4/kernel/vAdam/conv_transpose_4/bias/v*m
Tinf
d2b*
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
 __inference__traced_restore_8125??
?
?
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_7188

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_3_layer_call_fn_7278

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_55002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  @::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?	
?
@__inference_conv_1_layer_call_and_return_conditional_losses_6827

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
z
%__inference_conv_2_layer_call_fn_6993

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_2_layer_call_and_return_conditional_losses_53352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_5388

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@@@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_1_layer_call_fn_6900

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_52762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_1_layer_call_fn_6887

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_52582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
G
+__inference_leaky_relu_4_layer_call_fn_7362

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_4_layer_call_and_return_conditional_losses_55942
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_6_layer_call_and_return_conditional_losses_7456

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
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
?
?
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_4974

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_2_layer_call_fn_7057

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_46052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_3_layer_call_fn_7214

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_47092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_5005

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?_
?
?__inference_model_layer_call_and_return_conditional_losses_6095

inputs
conv_1_5999
conv_1_6001
batchnorm_1_6004
batchnorm_1_6006
batchnorm_1_6008
batchnorm_1_6010
conv_2_6014
conv_2_6016
batchnorm_2_6019
batchnorm_2_6021
batchnorm_2_6023
batchnorm_2_6025
conv_3_6029
conv_3_6031
batchnorm_3_6034
batchnorm_3_6036
batchnorm_3_6038
batchnorm_3_6040
conv_transpose_1_6044
conv_transpose_1_6046
batchnorm_4_6049
batchnorm_4_6051
batchnorm_4_6053
batchnorm_4_6055
conv_transpose_2_6059
conv_transpose_2_6061
batchnorm_5_6064
batchnorm_5_6066
batchnorm_5_6068
batchnorm_5_6070
conv_transpose_3_6074
conv_transpose_3_6076
batchnorm_6_6079
batchnorm_6_6081
batchnorm_6_6083
batchnorm_6_6085
conv_transpose_4_6089
conv_transpose_4_6091
identity??#batchnorm_1/StatefulPartitionedCall?#batchnorm_2/StatefulPartitionedCall?#batchnorm_3/StatefulPartitionedCall?#batchnorm_4/StatefulPartitionedCall?#batchnorm_5/StatefulPartitionedCall?#batchnorm_6/StatefulPartitionedCall?conv_1/StatefulPartitionedCall?conv_2/StatefulPartitionedCall?conv_3/StatefulPartitionedCall?(conv_transpose_1/StatefulPartitionedCall?(conv_transpose_2/StatefulPartitionedCall?(conv_transpose_3/StatefulPartitionedCall?(conv_transpose_4/StatefulPartitionedCall?
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_5999conv_1_6001*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_1_layer_call_and_return_conditional_losses_52232 
conv_1/StatefulPartitionedCall?
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batchnorm_1_6004batchnorm_1_6006batchnorm_1_6008batchnorm_1_6010*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_52762%
#batchnorm_1/StatefulPartitionedCall?
leaky_relu_1/PartitionedCallPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_1_layer_call_and_return_conditional_losses_53172
leaky_relu_1/PartitionedCall?
conv_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_1/PartitionedCall:output:0conv_2_6014conv_2_6016*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_2_layer_call_and_return_conditional_losses_53352 
conv_2/StatefulPartitionedCall?
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batchnorm_2_6019batchnorm_2_6021batchnorm_2_6023batchnorm_2_6025*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_53882%
#batchnorm_2/StatefulPartitionedCall?
leaky_relu_2/PartitionedCallPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_2_layer_call_and_return_conditional_losses_54292
leaky_relu_2/PartitionedCall?
conv_3/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_2/PartitionedCall:output:0conv_3_6029conv_3_6031*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_3_layer_call_and_return_conditional_losses_54472 
conv_3/StatefulPartitionedCall?
#batchnorm_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batchnorm_3_6034batchnorm_3_6036batchnorm_3_6038batchnorm_3_6040*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_55002%
#batchnorm_3/StatefulPartitionedCall?
leaky_relu_3/PartitionedCallPartitionedCall,batchnorm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_3_layer_call_and_return_conditional_losses_55412
leaky_relu_3/PartitionedCall?
(conv_transpose_1/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_3/PartitionedCall:output:0conv_transpose_1_6044conv_transpose_1_6046*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_1_layer_call_and_return_conditional_losses_47542*
(conv_transpose_1/StatefulPartitionedCall?
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_1/StatefulPartitionedCall:output:0batchnorm_4_6049batchnorm_4_6051batchnorm_4_6053batchnorm_4_6055*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_48572%
#batchnorm_4/StatefulPartitionedCall?
leaky_relu_4/PartitionedCallPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_4_layer_call_and_return_conditional_losses_55942
leaky_relu_4/PartitionedCall?
(conv_transpose_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_4/PartitionedCall:output:0conv_transpose_2_6059conv_transpose_2_6061*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_2_layer_call_and_return_conditional_losses_49022*
(conv_transpose_2/StatefulPartitionedCall?
#batchnorm_5/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_2/StatefulPartitionedCall:output:0batchnorm_5_6064batchnorm_5_6066batchnorm_5_6068batchnorm_5_6070*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_50052%
#batchnorm_5/StatefulPartitionedCall?
leaky_relu_5/PartitionedCallPartitionedCall,batchnorm_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_5_layer_call_and_return_conditional_losses_56472
leaky_relu_5/PartitionedCall?
(conv_transpose_3/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_5/PartitionedCall:output:0conv_transpose_3_6074conv_transpose_3_6076*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_3_layer_call_and_return_conditional_losses_50502*
(conv_transpose_3/StatefulPartitionedCall?
#batchnorm_6/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_3/StatefulPartitionedCall:output:0batchnorm_6_6079batchnorm_6_6081batchnorm_6_6083batchnorm_6_6085*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_6_layer_call_and_return_conditional_losses_51532%
#batchnorm_6/StatefulPartitionedCall?
leaky_relu_6/PartitionedCallPartitionedCall,batchnorm_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_6_layer_call_and_return_conditional_losses_57002
leaky_relu_6/PartitionedCall?
(conv_transpose_4/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_6/PartitionedCall:output:0conv_transpose_4_6089conv_transpose_4_6091*
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
GPU 2J 8? *S
fNRL
J__inference_conv_transpose_4_layer_call_and_return_conditional_losses_51992*
(conv_transpose_4/StatefulPartitionedCall?
IdentityIdentity1conv_transpose_4/StatefulPartitionedCall:output:0$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_3/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall$^batchnorm_5/StatefulPartitionedCall$^batchnorm_6/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall)^conv_transpose_1/StatefulPartitionedCall)^conv_transpose_2/StatefulPartitionedCall)^conv_transpose_3/StatefulPartitionedCall)^conv_transpose_4/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_3/StatefulPartitionedCall#batchnorm_3/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2J
#batchnorm_5/StatefulPartitionedCall#batchnorm_5/StatefulPartitionedCall2J
#batchnorm_6/StatefulPartitionedCall#batchnorm_6/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2T
(conv_transpose_1/StatefulPartitionedCall(conv_transpose_1/StatefulPartitionedCall2T
(conv_transpose_2/StatefulPartitionedCall(conv_transpose_2/StatefulPartitionedCall2T
(conv_transpose_3/StatefulPartitionedCall(conv_transpose_3/StatefulPartitionedCall2T
(conv_transpose_4/StatefulPartitionedCall(conv_transpose_4/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_7013

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_5370

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@@@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_7252

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
b
F__inference_leaky_relu_6_layer_call_and_return_conditional_losses_7505

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+??????????????????????????? *
alpha%???>2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_7326

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_2_layer_call_fn_7108

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_53702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@@@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
b
F__inference_leaky_relu_4_layer_call_and_return_conditional_losses_5594

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+???????????????????????????@*
alpha%???>2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?_
?
?__inference_model_layer_call_and_return_conditional_losses_5915

inputs
conv_1_5819
conv_1_5821
batchnorm_1_5824
batchnorm_1_5826
batchnorm_1_5828
batchnorm_1_5830
conv_2_5834
conv_2_5836
batchnorm_2_5839
batchnorm_2_5841
batchnorm_2_5843
batchnorm_2_5845
conv_3_5849
conv_3_5851
batchnorm_3_5854
batchnorm_3_5856
batchnorm_3_5858
batchnorm_3_5860
conv_transpose_1_5864
conv_transpose_1_5866
batchnorm_4_5869
batchnorm_4_5871
batchnorm_4_5873
batchnorm_4_5875
conv_transpose_2_5879
conv_transpose_2_5881
batchnorm_5_5884
batchnorm_5_5886
batchnorm_5_5888
batchnorm_5_5890
conv_transpose_3_5894
conv_transpose_3_5896
batchnorm_6_5899
batchnorm_6_5901
batchnorm_6_5903
batchnorm_6_5905
conv_transpose_4_5909
conv_transpose_4_5911
identity??#batchnorm_1/StatefulPartitionedCall?#batchnorm_2/StatefulPartitionedCall?#batchnorm_3/StatefulPartitionedCall?#batchnorm_4/StatefulPartitionedCall?#batchnorm_5/StatefulPartitionedCall?#batchnorm_6/StatefulPartitionedCall?conv_1/StatefulPartitionedCall?conv_2/StatefulPartitionedCall?conv_3/StatefulPartitionedCall?(conv_transpose_1/StatefulPartitionedCall?(conv_transpose_2/StatefulPartitionedCall?(conv_transpose_3/StatefulPartitionedCall?(conv_transpose_4/StatefulPartitionedCall?
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_5819conv_1_5821*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_1_layer_call_and_return_conditional_losses_52232 
conv_1/StatefulPartitionedCall?
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batchnorm_1_5824batchnorm_1_5826batchnorm_1_5828batchnorm_1_5830*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_52582%
#batchnorm_1/StatefulPartitionedCall?
leaky_relu_1/PartitionedCallPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_1_layer_call_and_return_conditional_losses_53172
leaky_relu_1/PartitionedCall?
conv_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_1/PartitionedCall:output:0conv_2_5834conv_2_5836*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_2_layer_call_and_return_conditional_losses_53352 
conv_2/StatefulPartitionedCall?
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batchnorm_2_5839batchnorm_2_5841batchnorm_2_5843batchnorm_2_5845*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_53702%
#batchnorm_2/StatefulPartitionedCall?
leaky_relu_2/PartitionedCallPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_2_layer_call_and_return_conditional_losses_54292
leaky_relu_2/PartitionedCall?
conv_3/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_2/PartitionedCall:output:0conv_3_5849conv_3_5851*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_3_layer_call_and_return_conditional_losses_54472 
conv_3/StatefulPartitionedCall?
#batchnorm_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batchnorm_3_5854batchnorm_3_5856batchnorm_3_5858batchnorm_3_5860*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_54822%
#batchnorm_3/StatefulPartitionedCall?
leaky_relu_3/PartitionedCallPartitionedCall,batchnorm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_3_layer_call_and_return_conditional_losses_55412
leaky_relu_3/PartitionedCall?
(conv_transpose_1/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_3/PartitionedCall:output:0conv_transpose_1_5864conv_transpose_1_5866*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_1_layer_call_and_return_conditional_losses_47542*
(conv_transpose_1/StatefulPartitionedCall?
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_1/StatefulPartitionedCall:output:0batchnorm_4_5869batchnorm_4_5871batchnorm_4_5873batchnorm_4_5875*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_48262%
#batchnorm_4/StatefulPartitionedCall?
leaky_relu_4/PartitionedCallPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_4_layer_call_and_return_conditional_losses_55942
leaky_relu_4/PartitionedCall?
(conv_transpose_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_4/PartitionedCall:output:0conv_transpose_2_5879conv_transpose_2_5881*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_2_layer_call_and_return_conditional_losses_49022*
(conv_transpose_2/StatefulPartitionedCall?
#batchnorm_5/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_2/StatefulPartitionedCall:output:0batchnorm_5_5884batchnorm_5_5886batchnorm_5_5888batchnorm_5_5890*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_49742%
#batchnorm_5/StatefulPartitionedCall?
leaky_relu_5/PartitionedCallPartitionedCall,batchnorm_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_5_layer_call_and_return_conditional_losses_56472
leaky_relu_5/PartitionedCall?
(conv_transpose_3/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_5/PartitionedCall:output:0conv_transpose_3_5894conv_transpose_3_5896*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_3_layer_call_and_return_conditional_losses_50502*
(conv_transpose_3/StatefulPartitionedCall?
#batchnorm_6/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_3/StatefulPartitionedCall:output:0batchnorm_6_5899batchnorm_6_5901batchnorm_6_5903batchnorm_6_5905*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_6_layer_call_and_return_conditional_losses_51222%
#batchnorm_6/StatefulPartitionedCall?
leaky_relu_6/PartitionedCallPartitionedCall,batchnorm_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_6_layer_call_and_return_conditional_losses_57002
leaky_relu_6/PartitionedCall?
(conv_transpose_4/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_6/PartitionedCall:output:0conv_transpose_4_5909conv_transpose_4_5911*
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
GPU 2J 8? *S
fNRL
J__inference_conv_transpose_4_layer_call_and_return_conditional_losses_51992*
(conv_transpose_4/StatefulPartitionedCall?
IdentityIdentity1conv_transpose_4/StatefulPartitionedCall:output:0$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_3/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall$^batchnorm_5/StatefulPartitionedCall$^batchnorm_6/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall)^conv_transpose_1/StatefulPartitionedCall)^conv_transpose_2/StatefulPartitionedCall)^conv_transpose_3/StatefulPartitionedCall)^conv_transpose_4/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_3/StatefulPartitionedCall#batchnorm_3/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2J
#batchnorm_5/StatefulPartitionedCall#batchnorm_5/StatefulPartitionedCall2J
#batchnorm_6/StatefulPartitionedCall#batchnorm_6/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2T
(conv_transpose_1/StatefulPartitionedCall(conv_transpose_1/StatefulPartitionedCall2T
(conv_transpose_2/StatefulPartitionedCall(conv_transpose_2/StatefulPartitionedCall2T
(conv_transpose_3/StatefulPartitionedCall(conv_transpose_3/StatefulPartitionedCall2T
(conv_transpose_4/StatefulPartitionedCall(conv_transpose_4/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_6856

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
b
F__inference_leaky_relu_3_layer_call_and_return_conditional_losses_7283

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????  @*
alpha%???>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_4_layer_call_fn_7339

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_48262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
@__inference_conv_1_layer_call_and_return_conditional_losses_5223

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_7095

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@@@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?#
?
J__inference_conv_transpose_2_layer_call_and_return_conditional_losses_4902

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_7031

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
@__inference_conv_2_layer_call_and_return_conditional_losses_5335

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_4709

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_6_layer_call_and_return_conditional_losses_5122

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
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
?
?
E__inference_batchnorm_6_layer_call_and_return_conditional_losses_5153

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_6174
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identity??StatefulPartitionedCall?
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
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_60952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameinput_layer
?
?
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_4678

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
G
+__inference_leaky_relu_3_layer_call_fn_7288

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
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_3_layer_call_and_return_conditional_losses_55412
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?$
?
J__inference_conv_transpose_4_layer_call_and_return_conditional_losses_5199

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_4470

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
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
?
?
*__inference_batchnorm_4_layer_call_fn_7352

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_48572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
b
F__inference_leaky_relu_6_layer_call_and_return_conditional_losses_5700

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+??????????????????????????? *
alpha%???>2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_5_layer_call_fn_7413

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_49742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_6_layer_call_fn_7487

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_6_layer_call_and_return_conditional_losses_51222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_6817

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identity??StatefulPartitionedCall?
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
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_60952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_4826

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_7170

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_5276

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_7234

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  @::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_2_layer_call_fn_7044

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_45742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_1_layer_call_fn_6964

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_45012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
/__inference_conv_transpose_3_layer_call_fn_5060

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_3_layer_call_and_return_conditional_losses_50502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
b
F__inference_leaky_relu_2_layer_call_and_return_conditional_losses_5429

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@@@*
alpha%???>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_4605

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_6_layer_call_fn_7500

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_6_layer_call_and_return_conditional_losses_51532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?
__inference__wrapped_model_4408
input_layer/
+model_conv_1_conv2d_readvariableop_resource0
,model_conv_1_biasadd_readvariableop_resource-
)model_batchnorm_1_readvariableop_resource/
+model_batchnorm_1_readvariableop_1_resource>
:model_batchnorm_1_fusedbatchnormv3_readvariableop_resource@
<model_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource/
+model_conv_2_conv2d_readvariableop_resource0
,model_conv_2_biasadd_readvariableop_resource-
)model_batchnorm_2_readvariableop_resource/
+model_batchnorm_2_readvariableop_1_resource>
:model_batchnorm_2_fusedbatchnormv3_readvariableop_resource@
<model_batchnorm_2_fusedbatchnormv3_readvariableop_1_resource/
+model_conv_3_conv2d_readvariableop_resource0
,model_conv_3_biasadd_readvariableop_resource-
)model_batchnorm_3_readvariableop_resource/
+model_batchnorm_3_readvariableop_1_resource>
:model_batchnorm_3_fusedbatchnormv3_readvariableop_resource@
<model_batchnorm_3_fusedbatchnormv3_readvariableop_1_resourceC
?model_conv_transpose_1_conv2d_transpose_readvariableop_resource:
6model_conv_transpose_1_biasadd_readvariableop_resource-
)model_batchnorm_4_readvariableop_resource/
+model_batchnorm_4_readvariableop_1_resource>
:model_batchnorm_4_fusedbatchnormv3_readvariableop_resource@
<model_batchnorm_4_fusedbatchnormv3_readvariableop_1_resourceC
?model_conv_transpose_2_conv2d_transpose_readvariableop_resource:
6model_conv_transpose_2_biasadd_readvariableop_resource-
)model_batchnorm_5_readvariableop_resource/
+model_batchnorm_5_readvariableop_1_resource>
:model_batchnorm_5_fusedbatchnormv3_readvariableop_resource@
<model_batchnorm_5_fusedbatchnormv3_readvariableop_1_resourceC
?model_conv_transpose_3_conv2d_transpose_readvariableop_resource:
6model_conv_transpose_3_biasadd_readvariableop_resource-
)model_batchnorm_6_readvariableop_resource/
+model_batchnorm_6_readvariableop_1_resource>
:model_batchnorm_6_fusedbatchnormv3_readvariableop_resource@
<model_batchnorm_6_fusedbatchnormv3_readvariableop_1_resourceC
?model_conv_transpose_4_conv2d_transpose_readvariableop_resource:
6model_conv_transpose_4_biasadd_readvariableop_resource
identity??1model/batchnorm_1/FusedBatchNormV3/ReadVariableOp?3model/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1? model/batchnorm_1/ReadVariableOp?"model/batchnorm_1/ReadVariableOp_1?1model/batchnorm_2/FusedBatchNormV3/ReadVariableOp?3model/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1? model/batchnorm_2/ReadVariableOp?"model/batchnorm_2/ReadVariableOp_1?1model/batchnorm_3/FusedBatchNormV3/ReadVariableOp?3model/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1? model/batchnorm_3/ReadVariableOp?"model/batchnorm_3/ReadVariableOp_1?1model/batchnorm_4/FusedBatchNormV3/ReadVariableOp?3model/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1? model/batchnorm_4/ReadVariableOp?"model/batchnorm_4/ReadVariableOp_1?1model/batchnorm_5/FusedBatchNormV3/ReadVariableOp?3model/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1? model/batchnorm_5/ReadVariableOp?"model/batchnorm_5/ReadVariableOp_1?1model/batchnorm_6/FusedBatchNormV3/ReadVariableOp?3model/batchnorm_6/FusedBatchNormV3/ReadVariableOp_1? model/batchnorm_6/ReadVariableOp?"model/batchnorm_6/ReadVariableOp_1?#model/conv_1/BiasAdd/ReadVariableOp?"model/conv_1/Conv2D/ReadVariableOp?#model/conv_2/BiasAdd/ReadVariableOp?"model/conv_2/Conv2D/ReadVariableOp?#model/conv_3/BiasAdd/ReadVariableOp?"model/conv_3/Conv2D/ReadVariableOp?-model/conv_transpose_1/BiasAdd/ReadVariableOp?6model/conv_transpose_1/conv2d_transpose/ReadVariableOp?-model/conv_transpose_2/BiasAdd/ReadVariableOp?6model/conv_transpose_2/conv2d_transpose/ReadVariableOp?-model/conv_transpose_3/BiasAdd/ReadVariableOp?6model/conv_transpose_3/conv2d_transpose/ReadVariableOp?-model/conv_transpose_4/BiasAdd/ReadVariableOp?6model/conv_transpose_4/conv2d_transpose/ReadVariableOp?
"model/conv_1/Conv2D/ReadVariableOpReadVariableOp+model_conv_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02$
"model/conv_1/Conv2D/ReadVariableOp?
model/conv_1/Conv2DConv2Dinput_layer*model/conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
model/conv_1/Conv2D?
#model/conv_1/BiasAdd/ReadVariableOpReadVariableOp,model_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#model/conv_1/BiasAdd/ReadVariableOp?
model/conv_1/BiasAddBiasAddmodel/conv_1/Conv2D:output:0+model/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
model/conv_1/BiasAdd?
 model/batchnorm_1/ReadVariableOpReadVariableOp)model_batchnorm_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 model/batchnorm_1/ReadVariableOp?
"model/batchnorm_1/ReadVariableOp_1ReadVariableOp+model_batchnorm_1_readvariableop_1_resource*
_output_shapes
: *
dtype02$
"model/batchnorm_1/ReadVariableOp_1?
1model/batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp:model_batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype023
1model/batchnorm_1/FusedBatchNormV3/ReadVariableOp?
3model/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<model_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype025
3model/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?
"model/batchnorm_1/FusedBatchNormV3FusedBatchNormV3model/conv_1/BiasAdd:output:0(model/batchnorm_1/ReadVariableOp:value:0*model/batchnorm_1/ReadVariableOp_1:value:09model/batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:0;model/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2$
"model/batchnorm_1/FusedBatchNormV3?
model/leaky_relu_1/LeakyRelu	LeakyRelu&model/batchnorm_1/FusedBatchNormV3:y:0*1
_output_shapes
:??????????? *
alpha%???>2
model/leaky_relu_1/LeakyRelu?
"model/conv_2/Conv2D/ReadVariableOpReadVariableOp+model_conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02$
"model/conv_2/Conv2D/ReadVariableOp?
model/conv_2/Conv2DConv2D*model/leaky_relu_1/LeakyRelu:activations:0*model/conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
model/conv_2/Conv2D?
#model/conv_2/BiasAdd/ReadVariableOpReadVariableOp,model_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#model/conv_2/BiasAdd/ReadVariableOp?
model/conv_2/BiasAddBiasAddmodel/conv_2/Conv2D:output:0+model/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2
model/conv_2/BiasAdd?
 model/batchnorm_2/ReadVariableOpReadVariableOp)model_batchnorm_2_readvariableop_resource*
_output_shapes
:@*
dtype02"
 model/batchnorm_2/ReadVariableOp?
"model/batchnorm_2/ReadVariableOp_1ReadVariableOp+model_batchnorm_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02$
"model/batchnorm_2/ReadVariableOp_1?
1model/batchnorm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp:model_batchnorm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype023
1model/batchnorm_2/FusedBatchNormV3/ReadVariableOp?
3model/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<model_batchnorm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3model/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1?
"model/batchnorm_2/FusedBatchNormV3FusedBatchNormV3model/conv_2/BiasAdd:output:0(model/batchnorm_2/ReadVariableOp:value:0*model/batchnorm_2/ReadVariableOp_1:value:09model/batchnorm_2/FusedBatchNormV3/ReadVariableOp:value:0;model/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( 2$
"model/batchnorm_2/FusedBatchNormV3?
model/leaky_relu_2/LeakyRelu	LeakyRelu&model/batchnorm_2/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@@*
alpha%???>2
model/leaky_relu_2/LeakyRelu?
"model/conv_3/Conv2D/ReadVariableOpReadVariableOp+model_conv_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"model/conv_3/Conv2D/ReadVariableOp?
model/conv_3/Conv2DConv2D*model/leaky_relu_2/LeakyRelu:activations:0*model/conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
model/conv_3/Conv2D?
#model/conv_3/BiasAdd/ReadVariableOpReadVariableOp,model_conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#model/conv_3/BiasAdd/ReadVariableOp?
model/conv_3/BiasAddBiasAddmodel/conv_3/Conv2D:output:0+model/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
model/conv_3/BiasAdd?
 model/batchnorm_3/ReadVariableOpReadVariableOp)model_batchnorm_3_readvariableop_resource*
_output_shapes
:@*
dtype02"
 model/batchnorm_3/ReadVariableOp?
"model/batchnorm_3/ReadVariableOp_1ReadVariableOp+model_batchnorm_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02$
"model/batchnorm_3/ReadVariableOp_1?
1model/batchnorm_3/FusedBatchNormV3/ReadVariableOpReadVariableOp:model_batchnorm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype023
1model/batchnorm_3/FusedBatchNormV3/ReadVariableOp?
3model/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<model_batchnorm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3model/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1?
"model/batchnorm_3/FusedBatchNormV3FusedBatchNormV3model/conv_3/BiasAdd:output:0(model/batchnorm_3/ReadVariableOp:value:0*model/batchnorm_3/ReadVariableOp_1:value:09model/batchnorm_3/FusedBatchNormV3/ReadVariableOp:value:0;model/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2$
"model/batchnorm_3/FusedBatchNormV3?
model/leaky_relu_3/LeakyRelu	LeakyRelu&model/batchnorm_3/FusedBatchNormV3:y:0*/
_output_shapes
:?????????  @*
alpha%???>2
model/leaky_relu_3/LeakyRelu?
model/conv_transpose_1/ShapeShape*model/leaky_relu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2
model/conv_transpose_1/Shape?
*model/conv_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model/conv_transpose_1/strided_slice/stack?
,model/conv_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_transpose_1/strided_slice/stack_1?
,model/conv_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_transpose_1/strided_slice/stack_2?
$model/conv_transpose_1/strided_sliceStridedSlice%model/conv_transpose_1/Shape:output:03model/conv_transpose_1/strided_slice/stack:output:05model/conv_transpose_1/strided_slice/stack_1:output:05model/conv_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$model/conv_transpose_1/strided_slice?
model/conv_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2 
model/conv_transpose_1/stack/1?
model/conv_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2 
model/conv_transpose_1/stack/2?
model/conv_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2 
model/conv_transpose_1/stack/3?
model/conv_transpose_1/stackPack-model/conv_transpose_1/strided_slice:output:0'model/conv_transpose_1/stack/1:output:0'model/conv_transpose_1/stack/2:output:0'model/conv_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
model/conv_transpose_1/stack?
,model/conv_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv_transpose_1/strided_slice_1/stack?
.model/conv_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv_transpose_1/strided_slice_1/stack_1?
.model/conv_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv_transpose_1/strided_slice_1/stack_2?
&model/conv_transpose_1/strided_slice_1StridedSlice%model/conv_transpose_1/stack:output:05model/conv_transpose_1/strided_slice_1/stack:output:07model/conv_transpose_1/strided_slice_1/stack_1:output:07model/conv_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/conv_transpose_1/strided_slice_1?
6model/conv_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp?model_conv_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype028
6model/conv_transpose_1/conv2d_transpose/ReadVariableOp?
'model/conv_transpose_1/conv2d_transposeConv2DBackpropInput%model/conv_transpose_1/stack:output:0>model/conv_transpose_1/conv2d_transpose/ReadVariableOp:value:0*model/leaky_relu_3/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2)
'model/conv_transpose_1/conv2d_transpose?
-model/conv_transpose_1/BiasAdd/ReadVariableOpReadVariableOp6model_conv_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model/conv_transpose_1/BiasAdd/ReadVariableOp?
model/conv_transpose_1/BiasAddBiasAdd0model/conv_transpose_1/conv2d_transpose:output:05model/conv_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2 
model/conv_transpose_1/BiasAdd?
 model/batchnorm_4/ReadVariableOpReadVariableOp)model_batchnorm_4_readvariableop_resource*
_output_shapes
:@*
dtype02"
 model/batchnorm_4/ReadVariableOp?
"model/batchnorm_4/ReadVariableOp_1ReadVariableOp+model_batchnorm_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02$
"model/batchnorm_4/ReadVariableOp_1?
1model/batchnorm_4/FusedBatchNormV3/ReadVariableOpReadVariableOp:model_batchnorm_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype023
1model/batchnorm_4/FusedBatchNormV3/ReadVariableOp?
3model/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<model_batchnorm_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3model/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1?
"model/batchnorm_4/FusedBatchNormV3FusedBatchNormV3'model/conv_transpose_1/BiasAdd:output:0(model/batchnorm_4/ReadVariableOp:value:0*model/batchnorm_4/ReadVariableOp_1:value:09model/batchnorm_4/FusedBatchNormV3/ReadVariableOp:value:0;model/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2$
"model/batchnorm_4/FusedBatchNormV3?
model/leaky_relu_4/LeakyRelu	LeakyRelu&model/batchnorm_4/FusedBatchNormV3:y:0*/
_output_shapes
:?????????  @*
alpha%???>2
model/leaky_relu_4/LeakyRelu?
model/conv_transpose_2/ShapeShape*model/leaky_relu_4/LeakyRelu:activations:0*
T0*
_output_shapes
:2
model/conv_transpose_2/Shape?
*model/conv_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model/conv_transpose_2/strided_slice/stack?
,model/conv_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_transpose_2/strided_slice/stack_1?
,model/conv_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_transpose_2/strided_slice/stack_2?
$model/conv_transpose_2/strided_sliceStridedSlice%model/conv_transpose_2/Shape:output:03model/conv_transpose_2/strided_slice/stack:output:05model/conv_transpose_2/strided_slice/stack_1:output:05model/conv_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$model/conv_transpose_2/strided_slice?
model/conv_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2 
model/conv_transpose_2/stack/1?
model/conv_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2 
model/conv_transpose_2/stack/2?
model/conv_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2 
model/conv_transpose_2/stack/3?
model/conv_transpose_2/stackPack-model/conv_transpose_2/strided_slice:output:0'model/conv_transpose_2/stack/1:output:0'model/conv_transpose_2/stack/2:output:0'model/conv_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
model/conv_transpose_2/stack?
,model/conv_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv_transpose_2/strided_slice_1/stack?
.model/conv_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv_transpose_2/strided_slice_1/stack_1?
.model/conv_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv_transpose_2/strided_slice_1/stack_2?
&model/conv_transpose_2/strided_slice_1StridedSlice%model/conv_transpose_2/stack:output:05model/conv_transpose_2/strided_slice_1/stack:output:07model/conv_transpose_2/strided_slice_1/stack_1:output:07model/conv_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/conv_transpose_2/strided_slice_1?
6model/conv_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp?model_conv_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype028
6model/conv_transpose_2/conv2d_transpose/ReadVariableOp?
'model/conv_transpose_2/conv2d_transposeConv2DBackpropInput%model/conv_transpose_2/stack:output:0>model/conv_transpose_2/conv2d_transpose/ReadVariableOp:value:0*model/leaky_relu_4/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2)
'model/conv_transpose_2/conv2d_transpose?
-model/conv_transpose_2/BiasAdd/ReadVariableOpReadVariableOp6model_conv_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model/conv_transpose_2/BiasAdd/ReadVariableOp?
model/conv_transpose_2/BiasAddBiasAdd0model/conv_transpose_2/conv2d_transpose:output:05model/conv_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2 
model/conv_transpose_2/BiasAdd?
 model/batchnorm_5/ReadVariableOpReadVariableOp)model_batchnorm_5_readvariableop_resource*
_output_shapes
:@*
dtype02"
 model/batchnorm_5/ReadVariableOp?
"model/batchnorm_5/ReadVariableOp_1ReadVariableOp+model_batchnorm_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02$
"model/batchnorm_5/ReadVariableOp_1?
1model/batchnorm_5/FusedBatchNormV3/ReadVariableOpReadVariableOp:model_batchnorm_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype023
1model/batchnorm_5/FusedBatchNormV3/ReadVariableOp?
3model/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<model_batchnorm_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3model/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1?
"model/batchnorm_5/FusedBatchNormV3FusedBatchNormV3'model/conv_transpose_2/BiasAdd:output:0(model/batchnorm_5/ReadVariableOp:value:0*model/batchnorm_5/ReadVariableOp_1:value:09model/batchnorm_5/FusedBatchNormV3/ReadVariableOp:value:0;model/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( 2$
"model/batchnorm_5/FusedBatchNormV3?
model/leaky_relu_5/LeakyRelu	LeakyRelu&model/batchnorm_5/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@@*
alpha%???>2
model/leaky_relu_5/LeakyRelu?
model/conv_transpose_3/ShapeShape*model/leaky_relu_5/LeakyRelu:activations:0*
T0*
_output_shapes
:2
model/conv_transpose_3/Shape?
*model/conv_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model/conv_transpose_3/strided_slice/stack?
,model/conv_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_transpose_3/strided_slice/stack_1?
,model/conv_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_transpose_3/strided_slice/stack_2?
$model/conv_transpose_3/strided_sliceStridedSlice%model/conv_transpose_3/Shape:output:03model/conv_transpose_3/strided_slice/stack:output:05model/conv_transpose_3/strided_slice/stack_1:output:05model/conv_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$model/conv_transpose_3/strided_slice?
model/conv_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2 
model/conv_transpose_3/stack/1?
model/conv_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2 
model/conv_transpose_3/stack/2?
model/conv_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2 
model/conv_transpose_3/stack/3?
model/conv_transpose_3/stackPack-model/conv_transpose_3/strided_slice:output:0'model/conv_transpose_3/stack/1:output:0'model/conv_transpose_3/stack/2:output:0'model/conv_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
model/conv_transpose_3/stack?
,model/conv_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv_transpose_3/strided_slice_1/stack?
.model/conv_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv_transpose_3/strided_slice_1/stack_1?
.model/conv_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv_transpose_3/strided_slice_1/stack_2?
&model/conv_transpose_3/strided_slice_1StridedSlice%model/conv_transpose_3/stack:output:05model/conv_transpose_3/strided_slice_1/stack:output:07model/conv_transpose_3/strided_slice_1/stack_1:output:07model/conv_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/conv_transpose_3/strided_slice_1?
6model/conv_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp?model_conv_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype028
6model/conv_transpose_3/conv2d_transpose/ReadVariableOp?
'model/conv_transpose_3/conv2d_transposeConv2DBackpropInput%model/conv_transpose_3/stack:output:0>model/conv_transpose_3/conv2d_transpose/ReadVariableOp:value:0*model/leaky_relu_5/LeakyRelu:activations:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2)
'model/conv_transpose_3/conv2d_transpose?
-model/conv_transpose_3/BiasAdd/ReadVariableOpReadVariableOp6model_conv_transpose_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-model/conv_transpose_3/BiasAdd/ReadVariableOp?
model/conv_transpose_3/BiasAddBiasAdd0model/conv_transpose_3/conv2d_transpose:output:05model/conv_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2 
model/conv_transpose_3/BiasAdd?
 model/batchnorm_6/ReadVariableOpReadVariableOp)model_batchnorm_6_readvariableop_resource*
_output_shapes
: *
dtype02"
 model/batchnorm_6/ReadVariableOp?
"model/batchnorm_6/ReadVariableOp_1ReadVariableOp+model_batchnorm_6_readvariableop_1_resource*
_output_shapes
: *
dtype02$
"model/batchnorm_6/ReadVariableOp_1?
1model/batchnorm_6/FusedBatchNormV3/ReadVariableOpReadVariableOp:model_batchnorm_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype023
1model/batchnorm_6/FusedBatchNormV3/ReadVariableOp?
3model/batchnorm_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<model_batchnorm_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype025
3model/batchnorm_6/FusedBatchNormV3/ReadVariableOp_1?
"model/batchnorm_6/FusedBatchNormV3FusedBatchNormV3'model/conv_transpose_3/BiasAdd:output:0(model/batchnorm_6/ReadVariableOp:value:0*model/batchnorm_6/ReadVariableOp_1:value:09model/batchnorm_6/FusedBatchNormV3/ReadVariableOp:value:0;model/batchnorm_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2$
"model/batchnorm_6/FusedBatchNormV3?
model/leaky_relu_6/LeakyRelu	LeakyRelu&model/batchnorm_6/FusedBatchNormV3:y:0*1
_output_shapes
:??????????? *
alpha%???>2
model/leaky_relu_6/LeakyRelu?
model/conv_transpose_4/ShapeShape*model/leaky_relu_6/LeakyRelu:activations:0*
T0*
_output_shapes
:2
model/conv_transpose_4/Shape?
*model/conv_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model/conv_transpose_4/strided_slice/stack?
,model/conv_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_transpose_4/strided_slice/stack_1?
,model/conv_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_transpose_4/strided_slice/stack_2?
$model/conv_transpose_4/strided_sliceStridedSlice%model/conv_transpose_4/Shape:output:03model/conv_transpose_4/strided_slice/stack:output:05model/conv_transpose_4/strided_slice/stack_1:output:05model/conv_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$model/conv_transpose_4/strided_slice?
model/conv_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2 
model/conv_transpose_4/stack/1?
model/conv_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2 
model/conv_transpose_4/stack/2?
model/conv_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2 
model/conv_transpose_4/stack/3?
model/conv_transpose_4/stackPack-model/conv_transpose_4/strided_slice:output:0'model/conv_transpose_4/stack/1:output:0'model/conv_transpose_4/stack/2:output:0'model/conv_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
model/conv_transpose_4/stack?
,model/conv_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv_transpose_4/strided_slice_1/stack?
.model/conv_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv_transpose_4/strided_slice_1/stack_1?
.model/conv_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv_transpose_4/strided_slice_1/stack_2?
&model/conv_transpose_4/strided_slice_1StridedSlice%model/conv_transpose_4/stack:output:05model/conv_transpose_4/strided_slice_1/stack:output:07model/conv_transpose_4/strided_slice_1/stack_1:output:07model/conv_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/conv_transpose_4/strided_slice_1?
6model/conv_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp?model_conv_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype028
6model/conv_transpose_4/conv2d_transpose/ReadVariableOp?
'model/conv_transpose_4/conv2d_transposeConv2DBackpropInput%model/conv_transpose_4/stack:output:0>model/conv_transpose_4/conv2d_transpose/ReadVariableOp:value:0*model/leaky_relu_6/LeakyRelu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2)
'model/conv_transpose_4/conv2d_transpose?
-model/conv_transpose_4/BiasAdd/ReadVariableOpReadVariableOp6model_conv_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-model/conv_transpose_4/BiasAdd/ReadVariableOp?
model/conv_transpose_4/BiasAddBiasAdd0model/conv_transpose_4/conv2d_transpose:output:05model/conv_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2 
model/conv_transpose_4/BiasAdd?
model/conv_transpose_4/SigmoidSigmoid'model/conv_transpose_4/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2 
model/conv_transpose_4/Sigmoid?
IdentityIdentity"model/conv_transpose_4/Sigmoid:y:02^model/batchnorm_1/FusedBatchNormV3/ReadVariableOp4^model/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1!^model/batchnorm_1/ReadVariableOp#^model/batchnorm_1/ReadVariableOp_12^model/batchnorm_2/FusedBatchNormV3/ReadVariableOp4^model/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1!^model/batchnorm_2/ReadVariableOp#^model/batchnorm_2/ReadVariableOp_12^model/batchnorm_3/FusedBatchNormV3/ReadVariableOp4^model/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1!^model/batchnorm_3/ReadVariableOp#^model/batchnorm_3/ReadVariableOp_12^model/batchnorm_4/FusedBatchNormV3/ReadVariableOp4^model/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1!^model/batchnorm_4/ReadVariableOp#^model/batchnorm_4/ReadVariableOp_12^model/batchnorm_5/FusedBatchNormV3/ReadVariableOp4^model/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1!^model/batchnorm_5/ReadVariableOp#^model/batchnorm_5/ReadVariableOp_12^model/batchnorm_6/FusedBatchNormV3/ReadVariableOp4^model/batchnorm_6/FusedBatchNormV3/ReadVariableOp_1!^model/batchnorm_6/ReadVariableOp#^model/batchnorm_6/ReadVariableOp_1$^model/conv_1/BiasAdd/ReadVariableOp#^model/conv_1/Conv2D/ReadVariableOp$^model/conv_2/BiasAdd/ReadVariableOp#^model/conv_2/Conv2D/ReadVariableOp$^model/conv_3/BiasAdd/ReadVariableOp#^model/conv_3/Conv2D/ReadVariableOp.^model/conv_transpose_1/BiasAdd/ReadVariableOp7^model/conv_transpose_1/conv2d_transpose/ReadVariableOp.^model/conv_transpose_2/BiasAdd/ReadVariableOp7^model/conv_transpose_2/conv2d_transpose/ReadVariableOp.^model/conv_transpose_3/BiasAdd/ReadVariableOp7^model/conv_transpose_3/conv2d_transpose/ReadVariableOp.^model/conv_transpose_4/BiasAdd/ReadVariableOp7^model/conv_transpose_4/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::2f
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
"model/batchnorm_3/ReadVariableOp_1"model/batchnorm_3/ReadVariableOp_12f
1model/batchnorm_4/FusedBatchNormV3/ReadVariableOp1model/batchnorm_4/FusedBatchNormV3/ReadVariableOp2j
3model/batchnorm_4/FusedBatchNormV3/ReadVariableOp_13model/batchnorm_4/FusedBatchNormV3/ReadVariableOp_12D
 model/batchnorm_4/ReadVariableOp model/batchnorm_4/ReadVariableOp2H
"model/batchnorm_4/ReadVariableOp_1"model/batchnorm_4/ReadVariableOp_12f
1model/batchnorm_5/FusedBatchNormV3/ReadVariableOp1model/batchnorm_5/FusedBatchNormV3/ReadVariableOp2j
3model/batchnorm_5/FusedBatchNormV3/ReadVariableOp_13model/batchnorm_5/FusedBatchNormV3/ReadVariableOp_12D
 model/batchnorm_5/ReadVariableOp model/batchnorm_5/ReadVariableOp2H
"model/batchnorm_5/ReadVariableOp_1"model/batchnorm_5/ReadVariableOp_12f
1model/batchnorm_6/FusedBatchNormV3/ReadVariableOp1model/batchnorm_6/FusedBatchNormV3/ReadVariableOp2j
3model/batchnorm_6/FusedBatchNormV3/ReadVariableOp_13model/batchnorm_6/FusedBatchNormV3/ReadVariableOp_12D
 model/batchnorm_6/ReadVariableOp model/batchnorm_6/ReadVariableOp2H
"model/batchnorm_6/ReadVariableOp_1"model/batchnorm_6/ReadVariableOp_12J
#model/conv_1/BiasAdd/ReadVariableOp#model/conv_1/BiasAdd/ReadVariableOp2H
"model/conv_1/Conv2D/ReadVariableOp"model/conv_1/Conv2D/ReadVariableOp2J
#model/conv_2/BiasAdd/ReadVariableOp#model/conv_2/BiasAdd/ReadVariableOp2H
"model/conv_2/Conv2D/ReadVariableOp"model/conv_2/Conv2D/ReadVariableOp2J
#model/conv_3/BiasAdd/ReadVariableOp#model/conv_3/BiasAdd/ReadVariableOp2H
"model/conv_3/Conv2D/ReadVariableOp"model/conv_3/Conv2D/ReadVariableOp2^
-model/conv_transpose_1/BiasAdd/ReadVariableOp-model/conv_transpose_1/BiasAdd/ReadVariableOp2p
6model/conv_transpose_1/conv2d_transpose/ReadVariableOp6model/conv_transpose_1/conv2d_transpose/ReadVariableOp2^
-model/conv_transpose_2/BiasAdd/ReadVariableOp-model/conv_transpose_2/BiasAdd/ReadVariableOp2p
6model/conv_transpose_2/conv2d_transpose/ReadVariableOp6model/conv_transpose_2/conv2d_transpose/ReadVariableOp2^
-model/conv_transpose_3/BiasAdd/ReadVariableOp-model/conv_transpose_3/BiasAdd/ReadVariableOp2p
6model/conv_transpose_3/conv2d_transpose/ReadVariableOp6model/conv_transpose_3/conv2d_transpose/ReadVariableOp2^
-model/conv_transpose_4/BiasAdd/ReadVariableOp-model/conv_transpose_4/BiasAdd/ReadVariableOp2p
6model/conv_transpose_4/conv2d_transpose/ReadVariableOp6model/conv_transpose_4/conv2d_transpose/ReadVariableOp:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameinput_layer
?
b
F__inference_leaky_relu_1_layer_call_and_return_conditional_losses_6969

inputs
identityn
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:??????????? *
alpha%???>2
	LeakyReluu
IdentityIdentityLeakyRelu:activations:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
/__inference_conv_transpose_4_layer_call_fn_5209

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
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
GPU 2J 8? *S
fNRL
J__inference_conv_transpose_4_layer_call_and_return_conditional_losses_51992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?_
?
?__inference_model_layer_call_and_return_conditional_losses_5714
input_layer
conv_1_5234
conv_1_5236
batchnorm_1_5303
batchnorm_1_5305
batchnorm_1_5307
batchnorm_1_5309
conv_2_5346
conv_2_5348
batchnorm_2_5415
batchnorm_2_5417
batchnorm_2_5419
batchnorm_2_5421
conv_3_5458
conv_3_5460
batchnorm_3_5527
batchnorm_3_5529
batchnorm_3_5531
batchnorm_3_5533
conv_transpose_1_5549
conv_transpose_1_5551
batchnorm_4_5580
batchnorm_4_5582
batchnorm_4_5584
batchnorm_4_5586
conv_transpose_2_5602
conv_transpose_2_5604
batchnorm_5_5633
batchnorm_5_5635
batchnorm_5_5637
batchnorm_5_5639
conv_transpose_3_5655
conv_transpose_3_5657
batchnorm_6_5686
batchnorm_6_5688
batchnorm_6_5690
batchnorm_6_5692
conv_transpose_4_5708
conv_transpose_4_5710
identity??#batchnorm_1/StatefulPartitionedCall?#batchnorm_2/StatefulPartitionedCall?#batchnorm_3/StatefulPartitionedCall?#batchnorm_4/StatefulPartitionedCall?#batchnorm_5/StatefulPartitionedCall?#batchnorm_6/StatefulPartitionedCall?conv_1/StatefulPartitionedCall?conv_2/StatefulPartitionedCall?conv_3/StatefulPartitionedCall?(conv_transpose_1/StatefulPartitionedCall?(conv_transpose_2/StatefulPartitionedCall?(conv_transpose_3/StatefulPartitionedCall?(conv_transpose_4/StatefulPartitionedCall?
conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv_1_5234conv_1_5236*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_1_layer_call_and_return_conditional_losses_52232 
conv_1/StatefulPartitionedCall?
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batchnorm_1_5303batchnorm_1_5305batchnorm_1_5307batchnorm_1_5309*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_52582%
#batchnorm_1/StatefulPartitionedCall?
leaky_relu_1/PartitionedCallPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_1_layer_call_and_return_conditional_losses_53172
leaky_relu_1/PartitionedCall?
conv_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_1/PartitionedCall:output:0conv_2_5346conv_2_5348*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_2_layer_call_and_return_conditional_losses_53352 
conv_2/StatefulPartitionedCall?
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batchnorm_2_5415batchnorm_2_5417batchnorm_2_5419batchnorm_2_5421*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_53702%
#batchnorm_2/StatefulPartitionedCall?
leaky_relu_2/PartitionedCallPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_2_layer_call_and_return_conditional_losses_54292
leaky_relu_2/PartitionedCall?
conv_3/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_2/PartitionedCall:output:0conv_3_5458conv_3_5460*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_3_layer_call_and_return_conditional_losses_54472 
conv_3/StatefulPartitionedCall?
#batchnorm_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batchnorm_3_5527batchnorm_3_5529batchnorm_3_5531batchnorm_3_5533*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_54822%
#batchnorm_3/StatefulPartitionedCall?
leaky_relu_3/PartitionedCallPartitionedCall,batchnorm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_3_layer_call_and_return_conditional_losses_55412
leaky_relu_3/PartitionedCall?
(conv_transpose_1/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_3/PartitionedCall:output:0conv_transpose_1_5549conv_transpose_1_5551*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_1_layer_call_and_return_conditional_losses_47542*
(conv_transpose_1/StatefulPartitionedCall?
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_1/StatefulPartitionedCall:output:0batchnorm_4_5580batchnorm_4_5582batchnorm_4_5584batchnorm_4_5586*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_48262%
#batchnorm_4/StatefulPartitionedCall?
leaky_relu_4/PartitionedCallPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_4_layer_call_and_return_conditional_losses_55942
leaky_relu_4/PartitionedCall?
(conv_transpose_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_4/PartitionedCall:output:0conv_transpose_2_5602conv_transpose_2_5604*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_2_layer_call_and_return_conditional_losses_49022*
(conv_transpose_2/StatefulPartitionedCall?
#batchnorm_5/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_2/StatefulPartitionedCall:output:0batchnorm_5_5633batchnorm_5_5635batchnorm_5_5637batchnorm_5_5639*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_49742%
#batchnorm_5/StatefulPartitionedCall?
leaky_relu_5/PartitionedCallPartitionedCall,batchnorm_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_5_layer_call_and_return_conditional_losses_56472
leaky_relu_5/PartitionedCall?
(conv_transpose_3/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_5/PartitionedCall:output:0conv_transpose_3_5655conv_transpose_3_5657*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_3_layer_call_and_return_conditional_losses_50502*
(conv_transpose_3/StatefulPartitionedCall?
#batchnorm_6/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_3/StatefulPartitionedCall:output:0batchnorm_6_5686batchnorm_6_5688batchnorm_6_5690batchnorm_6_5692*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_6_layer_call_and_return_conditional_losses_51222%
#batchnorm_6/StatefulPartitionedCall?
leaky_relu_6/PartitionedCallPartitionedCall,batchnorm_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_6_layer_call_and_return_conditional_losses_57002
leaky_relu_6/PartitionedCall?
(conv_transpose_4/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_6/PartitionedCall:output:0conv_transpose_4_5708conv_transpose_4_5710*
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
GPU 2J 8? *S
fNRL
J__inference_conv_transpose_4_layer_call_and_return_conditional_losses_51992*
(conv_transpose_4/StatefulPartitionedCall?
IdentityIdentity1conv_transpose_4/StatefulPartitionedCall:output:0$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_3/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall$^batchnorm_5/StatefulPartitionedCall$^batchnorm_6/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall)^conv_transpose_1/StatefulPartitionedCall)^conv_transpose_2/StatefulPartitionedCall)^conv_transpose_3/StatefulPartitionedCall)^conv_transpose_4/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_3/StatefulPartitionedCall#batchnorm_3/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2J
#batchnorm_5/StatefulPartitionedCall#batchnorm_5/StatefulPartitionedCall2J
#batchnorm_6/StatefulPartitionedCall#batchnorm_6/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2T
(conv_transpose_1/StatefulPartitionedCall(conv_transpose_1/StatefulPartitionedCall2T
(conv_transpose_2/StatefulPartitionedCall(conv_transpose_2/StatefulPartitionedCall2T
(conv_transpose_3/StatefulPartitionedCall(conv_transpose_3/StatefulPartitionedCall2T
(conv_transpose_4/StatefulPartitionedCall(conv_transpose_4/StatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameinput_layer
?
?
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_4857

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
b
F__inference_leaky_relu_1_layer_call_and_return_conditional_losses_5317

inputs
identityn
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:??????????? *
alpha%???>2
	LeakyReluu
IdentityIdentityLeakyRelu:activations:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?_
?
?__inference_model_layer_call_and_return_conditional_losses_5813
input_layer
conv_1_5717
conv_1_5719
batchnorm_1_5722
batchnorm_1_5724
batchnorm_1_5726
batchnorm_1_5728
conv_2_5732
conv_2_5734
batchnorm_2_5737
batchnorm_2_5739
batchnorm_2_5741
batchnorm_2_5743
conv_3_5747
conv_3_5749
batchnorm_3_5752
batchnorm_3_5754
batchnorm_3_5756
batchnorm_3_5758
conv_transpose_1_5762
conv_transpose_1_5764
batchnorm_4_5767
batchnorm_4_5769
batchnorm_4_5771
batchnorm_4_5773
conv_transpose_2_5777
conv_transpose_2_5779
batchnorm_5_5782
batchnorm_5_5784
batchnorm_5_5786
batchnorm_5_5788
conv_transpose_3_5792
conv_transpose_3_5794
batchnorm_6_5797
batchnorm_6_5799
batchnorm_6_5801
batchnorm_6_5803
conv_transpose_4_5807
conv_transpose_4_5809
identity??#batchnorm_1/StatefulPartitionedCall?#batchnorm_2/StatefulPartitionedCall?#batchnorm_3/StatefulPartitionedCall?#batchnorm_4/StatefulPartitionedCall?#batchnorm_5/StatefulPartitionedCall?#batchnorm_6/StatefulPartitionedCall?conv_1/StatefulPartitionedCall?conv_2/StatefulPartitionedCall?conv_3/StatefulPartitionedCall?(conv_transpose_1/StatefulPartitionedCall?(conv_transpose_2/StatefulPartitionedCall?(conv_transpose_3/StatefulPartitionedCall?(conv_transpose_4/StatefulPartitionedCall?
conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv_1_5717conv_1_5719*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_1_layer_call_and_return_conditional_losses_52232 
conv_1/StatefulPartitionedCall?
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batchnorm_1_5722batchnorm_1_5724batchnorm_1_5726batchnorm_1_5728*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_52762%
#batchnorm_1/StatefulPartitionedCall?
leaky_relu_1/PartitionedCallPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_1_layer_call_and_return_conditional_losses_53172
leaky_relu_1/PartitionedCall?
conv_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_1/PartitionedCall:output:0conv_2_5732conv_2_5734*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_2_layer_call_and_return_conditional_losses_53352 
conv_2/StatefulPartitionedCall?
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batchnorm_2_5737batchnorm_2_5739batchnorm_2_5741batchnorm_2_5743*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_53882%
#batchnorm_2/StatefulPartitionedCall?
leaky_relu_2/PartitionedCallPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_2_layer_call_and_return_conditional_losses_54292
leaky_relu_2/PartitionedCall?
conv_3/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_2/PartitionedCall:output:0conv_3_5747conv_3_5749*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_3_layer_call_and_return_conditional_losses_54472 
conv_3/StatefulPartitionedCall?
#batchnorm_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batchnorm_3_5752batchnorm_3_5754batchnorm_3_5756batchnorm_3_5758*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_55002%
#batchnorm_3/StatefulPartitionedCall?
leaky_relu_3/PartitionedCallPartitionedCall,batchnorm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_3_layer_call_and_return_conditional_losses_55412
leaky_relu_3/PartitionedCall?
(conv_transpose_1/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_3/PartitionedCall:output:0conv_transpose_1_5762conv_transpose_1_5764*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_1_layer_call_and_return_conditional_losses_47542*
(conv_transpose_1/StatefulPartitionedCall?
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_1/StatefulPartitionedCall:output:0batchnorm_4_5767batchnorm_4_5769batchnorm_4_5771batchnorm_4_5773*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_48572%
#batchnorm_4/StatefulPartitionedCall?
leaky_relu_4/PartitionedCallPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_4_layer_call_and_return_conditional_losses_55942
leaky_relu_4/PartitionedCall?
(conv_transpose_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_4/PartitionedCall:output:0conv_transpose_2_5777conv_transpose_2_5779*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_2_layer_call_and_return_conditional_losses_49022*
(conv_transpose_2/StatefulPartitionedCall?
#batchnorm_5/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_2/StatefulPartitionedCall:output:0batchnorm_5_5782batchnorm_5_5784batchnorm_5_5786batchnorm_5_5788*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_50052%
#batchnorm_5/StatefulPartitionedCall?
leaky_relu_5/PartitionedCallPartitionedCall,batchnorm_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_5_layer_call_and_return_conditional_losses_56472
leaky_relu_5/PartitionedCall?
(conv_transpose_3/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_5/PartitionedCall:output:0conv_transpose_3_5792conv_transpose_3_5794*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_3_layer_call_and_return_conditional_losses_50502*
(conv_transpose_3/StatefulPartitionedCall?
#batchnorm_6/StatefulPartitionedCallStatefulPartitionedCall1conv_transpose_3/StatefulPartitionedCall:output:0batchnorm_6_5797batchnorm_6_5799batchnorm_6_5801batchnorm_6_5803*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_6_layer_call_and_return_conditional_losses_51532%
#batchnorm_6/StatefulPartitionedCall?
leaky_relu_6/PartitionedCallPartitionedCall,batchnorm_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_6_layer_call_and_return_conditional_losses_57002
leaky_relu_6/PartitionedCall?
(conv_transpose_4/StatefulPartitionedCallStatefulPartitionedCall%leaky_relu_6/PartitionedCall:output:0conv_transpose_4_5807conv_transpose_4_5809*
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
GPU 2J 8? *S
fNRL
J__inference_conv_transpose_4_layer_call_and_return_conditional_losses_51992*
(conv_transpose_4/StatefulPartitionedCall?
IdentityIdentity1conv_transpose_4/StatefulPartitionedCall:output:0$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_3/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall$^batchnorm_5/StatefulPartitionedCall$^batchnorm_6/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall)^conv_transpose_1/StatefulPartitionedCall)^conv_transpose_2/StatefulPartitionedCall)^conv_transpose_3/StatefulPartitionedCall)^conv_transpose_4/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_3/StatefulPartitionedCall#batchnorm_3/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2J
#batchnorm_5/StatefulPartitionedCall#batchnorm_5/StatefulPartitionedCall2J
#batchnorm_6/StatefulPartitionedCall#batchnorm_6/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2T
(conv_transpose_1/StatefulPartitionedCall(conv_transpose_1/StatefulPartitionedCall2T
(conv_transpose_2/StatefulPartitionedCall(conv_transpose_2/StatefulPartitionedCall2T
(conv_transpose_3/StatefulPartitionedCall(conv_transpose_3/StatefulPartitionedCall2T
(conv_transpose_4/StatefulPartitionedCall(conv_transpose_4/StatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameinput_layer
?	
?
@__inference_conv_2_layer_call_and_return_conditional_losses_6984

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:??????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
b
F__inference_leaky_relu_4_layer_call_and_return_conditional_losses_7357

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+???????????????????????????@*
alpha%???>2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
@__inference_conv_3_layer_call_and_return_conditional_losses_5447

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_5_layer_call_fn_7426

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_50052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_3_layer_call_fn_7265

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_54822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  @::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_5258

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_7382

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
G
+__inference_leaky_relu_2_layer_call_fn_7131

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
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_2_layer_call_and_return_conditional_losses_54292
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
/__inference_conv_transpose_2_layer_call_fn_4912

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_2_layer_call_and_return_conditional_losses_49022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_7077

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@@@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?	
?
@__inference_conv_3_layer_call_and_return_conditional_losses_7141

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_7400

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_6265
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identity??StatefulPartitionedCall?
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
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_44082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameinput_layer
?
?
*__inference_batchnorm_1_layer_call_fn_6951

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_44702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_3_layer_call_fn_7201

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_46782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
z
%__inference_conv_3_layer_call_fn_7150

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_3_layer_call_and_return_conditional_losses_54472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_5482

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  @::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
/__inference_conv_transpose_1_layer_call_fn_4764

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_conv_transpose_1_layer_call_and_return_conditional_losses_47542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_5994
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identity??StatefulPartitionedCall?
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
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*<
_read_only_resource_inputs
	
 !"%&*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_59152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameinput_layer
?
?
$__inference_model_layer_call_fn_6736

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identity??StatefulPartitionedCall?
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
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*<
_read_only_resource_inputs
	
 !"%&*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_59152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_6466

inputs)
%conv_1_conv2d_readvariableop_resource*
&conv_1_biasadd_readvariableop_resource'
#batchnorm_1_readvariableop_resource)
%batchnorm_1_readvariableop_1_resource8
4batchnorm_1_fusedbatchnormv3_readvariableop_resource:
6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource)
%conv_2_conv2d_readvariableop_resource*
&conv_2_biasadd_readvariableop_resource'
#batchnorm_2_readvariableop_resource)
%batchnorm_2_readvariableop_1_resource8
4batchnorm_2_fusedbatchnormv3_readvariableop_resource:
6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource)
%conv_3_conv2d_readvariableop_resource*
&conv_3_biasadd_readvariableop_resource'
#batchnorm_3_readvariableop_resource)
%batchnorm_3_readvariableop_1_resource8
4batchnorm_3_fusedbatchnormv3_readvariableop_resource:
6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource=
9conv_transpose_1_conv2d_transpose_readvariableop_resource4
0conv_transpose_1_biasadd_readvariableop_resource'
#batchnorm_4_readvariableop_resource)
%batchnorm_4_readvariableop_1_resource8
4batchnorm_4_fusedbatchnormv3_readvariableop_resource:
6batchnorm_4_fusedbatchnormv3_readvariableop_1_resource=
9conv_transpose_2_conv2d_transpose_readvariableop_resource4
0conv_transpose_2_biasadd_readvariableop_resource'
#batchnorm_5_readvariableop_resource)
%batchnorm_5_readvariableop_1_resource8
4batchnorm_5_fusedbatchnormv3_readvariableop_resource:
6batchnorm_5_fusedbatchnormv3_readvariableop_1_resource=
9conv_transpose_3_conv2d_transpose_readvariableop_resource4
0conv_transpose_3_biasadd_readvariableop_resource'
#batchnorm_6_readvariableop_resource)
%batchnorm_6_readvariableop_1_resource8
4batchnorm_6_fusedbatchnormv3_readvariableop_resource:
6batchnorm_6_fusedbatchnormv3_readvariableop_1_resource=
9conv_transpose_4_conv2d_transpose_readvariableop_resource4
0conv_transpose_4_biasadd_readvariableop_resource
identity??batchnorm_1/AssignNewValue?batchnorm_1/AssignNewValue_1?+batchnorm_1/FusedBatchNormV3/ReadVariableOp?-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?batchnorm_1/ReadVariableOp?batchnorm_1/ReadVariableOp_1?batchnorm_2/AssignNewValue?batchnorm_2/AssignNewValue_1?+batchnorm_2/FusedBatchNormV3/ReadVariableOp?-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1?batchnorm_2/ReadVariableOp?batchnorm_2/ReadVariableOp_1?batchnorm_3/AssignNewValue?batchnorm_3/AssignNewValue_1?+batchnorm_3/FusedBatchNormV3/ReadVariableOp?-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1?batchnorm_3/ReadVariableOp?batchnorm_3/ReadVariableOp_1?batchnorm_4/AssignNewValue?batchnorm_4/AssignNewValue_1?+batchnorm_4/FusedBatchNormV3/ReadVariableOp?-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1?batchnorm_4/ReadVariableOp?batchnorm_4/ReadVariableOp_1?batchnorm_5/AssignNewValue?batchnorm_5/AssignNewValue_1?+batchnorm_5/FusedBatchNormV3/ReadVariableOp?-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1?batchnorm_5/ReadVariableOp?batchnorm_5/ReadVariableOp_1?batchnorm_6/AssignNewValue?batchnorm_6/AssignNewValue_1?+batchnorm_6/FusedBatchNormV3/ReadVariableOp?-batchnorm_6/FusedBatchNormV3/ReadVariableOp_1?batchnorm_6/ReadVariableOp?batchnorm_6/ReadVariableOp_1?conv_1/BiasAdd/ReadVariableOp?conv_1/Conv2D/ReadVariableOp?conv_2/BiasAdd/ReadVariableOp?conv_2/Conv2D/ReadVariableOp?conv_3/BiasAdd/ReadVariableOp?conv_3/Conv2D/ReadVariableOp?'conv_transpose_1/BiasAdd/ReadVariableOp?0conv_transpose_1/conv2d_transpose/ReadVariableOp?'conv_transpose_2/BiasAdd/ReadVariableOp?0conv_transpose_2/conv2d_transpose/ReadVariableOp?'conv_transpose_3/BiasAdd/ReadVariableOp?0conv_transpose_3/conv2d_transpose/ReadVariableOp?'conv_transpose_4/BiasAdd/ReadVariableOp?0conv_transpose_4/conv2d_transpose/ReadVariableOp?
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv_1/Conv2D/ReadVariableOp?
conv_1/Conv2DConv2Dinputs$conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv_1/Conv2D?
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv_1/BiasAdd/ReadVariableOp?
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv_1/BiasAdd?
batchnorm_1/ReadVariableOpReadVariableOp#batchnorm_1_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm_1/ReadVariableOp?
batchnorm_1/ReadVariableOp_1ReadVariableOp%batchnorm_1_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm_1/ReadVariableOp_1?
+batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02-
+batchnorm_1/FusedBatchNormV3/ReadVariableOp?
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02/
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_1/FusedBatchNormV3FusedBatchNormV3conv_1/BiasAdd:output:0"batchnorm_1/ReadVariableOp:value:0$batchnorm_1/ReadVariableOp_1:value:03batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
batchnorm_1/FusedBatchNormV3?
batchnorm_1/AssignNewValueAssignVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource)batchnorm_1/FusedBatchNormV3:batch_mean:0,^batchnorm_1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*G
_class=
;9loc:@batchnorm_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batchnorm_1/AssignNewValue?
batchnorm_1/AssignNewValue_1AssignVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource-batchnorm_1/FusedBatchNormV3:batch_variance:0.^batchnorm_1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*I
_class?
=;loc:@batchnorm_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batchnorm_1/AssignNewValue_1?
leaky_relu_1/LeakyRelu	LeakyRelu batchnorm_1/FusedBatchNormV3:y:0*1
_output_shapes
:??????????? *
alpha%???>2
leaky_relu_1/LeakyRelu?
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv_2/Conv2D/ReadVariableOp?
conv_2/Conv2DConv2D$leaky_relu_1/LeakyRelu:activations:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
conv_2/Conv2D?
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_2/BiasAdd/ReadVariableOp?
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2
conv_2/BiasAdd?
batchnorm_2/ReadVariableOpReadVariableOp#batchnorm_2_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm_2/ReadVariableOp?
batchnorm_2/ReadVariableOp_1ReadVariableOp%batchnorm_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm_2/ReadVariableOp_1?
+batchnorm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batchnorm_2/FusedBatchNormV3/ReadVariableOp?
-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_2/FusedBatchNormV3FusedBatchNormV3conv_2/BiasAdd:output:0"batchnorm_2/ReadVariableOp:value:0$batchnorm_2/ReadVariableOp_1:value:03batchnorm_2/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
batchnorm_2/FusedBatchNormV3?
batchnorm_2/AssignNewValueAssignVariableOp4batchnorm_2_fusedbatchnormv3_readvariableop_resource)batchnorm_2/FusedBatchNormV3:batch_mean:0,^batchnorm_2/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*G
_class=
;9loc:@batchnorm_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batchnorm_2/AssignNewValue?
batchnorm_2/AssignNewValue_1AssignVariableOp6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource-batchnorm_2/FusedBatchNormV3:batch_variance:0.^batchnorm_2/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*I
_class?
=;loc:@batchnorm_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batchnorm_2/AssignNewValue_1?
leaky_relu_2/LeakyRelu	LeakyRelu batchnorm_2/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@@*
alpha%???>2
leaky_relu_2/LeakyRelu?
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv_3/Conv2D/ReadVariableOp?
conv_3/Conv2DConv2D$leaky_relu_2/LeakyRelu:activations:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
conv_3/Conv2D?
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_3/BiasAdd/ReadVariableOp?
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
conv_3/BiasAdd?
batchnorm_3/ReadVariableOpReadVariableOp#batchnorm_3_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm_3/ReadVariableOp?
batchnorm_3/ReadVariableOp_1ReadVariableOp%batchnorm_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm_3/ReadVariableOp_1?
+batchnorm_3/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batchnorm_3/FusedBatchNormV3/ReadVariableOp?
-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_3/FusedBatchNormV3FusedBatchNormV3conv_3/BiasAdd:output:0"batchnorm_3/ReadVariableOp:value:0$batchnorm_3/ReadVariableOp_1:value:03batchnorm_3/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
batchnorm_3/FusedBatchNormV3?
batchnorm_3/AssignNewValueAssignVariableOp4batchnorm_3_fusedbatchnormv3_readvariableop_resource)batchnorm_3/FusedBatchNormV3:batch_mean:0,^batchnorm_3/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*G
_class=
;9loc:@batchnorm_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batchnorm_3/AssignNewValue?
batchnorm_3/AssignNewValue_1AssignVariableOp6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource-batchnorm_3/FusedBatchNormV3:batch_variance:0.^batchnorm_3/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*I
_class?
=;loc:@batchnorm_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batchnorm_3/AssignNewValue_1?
leaky_relu_3/LeakyRelu	LeakyRelu batchnorm_3/FusedBatchNormV3:y:0*/
_output_shapes
:?????????  @*
alpha%???>2
leaky_relu_3/LeakyRelu?
conv_transpose_1/ShapeShape$leaky_relu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv_transpose_1/Shape?
$conv_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_1/strided_slice/stack?
&conv_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_1?
&conv_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_2?
conv_transpose_1/strided_sliceStridedSliceconv_transpose_1/Shape:output:0-conv_transpose_1/strided_slice/stack:output:0/conv_transpose_1/strided_slice/stack_1:output:0/conv_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_1/strided_slicev
conv_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_1/stack/1v
conv_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_1/stack/2v
conv_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_1/stack/3?
conv_transpose_1/stackPack'conv_transpose_1/strided_slice:output:0!conv_transpose_1/stack/1:output:0!conv_transpose_1/stack/2:output:0!conv_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_1/stack?
&conv_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_1/strided_slice_1/stack?
(conv_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_1?
(conv_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_2?
 conv_transpose_1/strided_slice_1StridedSliceconv_transpose_1/stack:output:0/conv_transpose_1/strided_slice_1/stack:output:01conv_transpose_1/strided_slice_1/stack_1:output:01conv_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_1/strided_slice_1?
0conv_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0conv_transpose_1/conv2d_transpose/ReadVariableOp?
!conv_transpose_1/conv2d_transposeConv2DBackpropInputconv_transpose_1/stack:output:08conv_transpose_1/conv2d_transpose/ReadVariableOp:value:0$leaky_relu_3/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2#
!conv_transpose_1/conv2d_transpose?
'conv_transpose_1/BiasAdd/ReadVariableOpReadVariableOp0conv_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'conv_transpose_1/BiasAdd/ReadVariableOp?
conv_transpose_1/BiasAddBiasAdd*conv_transpose_1/conv2d_transpose:output:0/conv_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
conv_transpose_1/BiasAdd?
batchnorm_4/ReadVariableOpReadVariableOp#batchnorm_4_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm_4/ReadVariableOp?
batchnorm_4/ReadVariableOp_1ReadVariableOp%batchnorm_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm_4/ReadVariableOp_1?
+batchnorm_4/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batchnorm_4/FusedBatchNormV3/ReadVariableOp?
-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_4/FusedBatchNormV3FusedBatchNormV3!conv_transpose_1/BiasAdd:output:0"batchnorm_4/ReadVariableOp:value:0$batchnorm_4/ReadVariableOp_1:value:03batchnorm_4/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
batchnorm_4/FusedBatchNormV3?
batchnorm_4/AssignNewValueAssignVariableOp4batchnorm_4_fusedbatchnormv3_readvariableop_resource)batchnorm_4/FusedBatchNormV3:batch_mean:0,^batchnorm_4/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*G
_class=
;9loc:@batchnorm_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batchnorm_4/AssignNewValue?
batchnorm_4/AssignNewValue_1AssignVariableOp6batchnorm_4_fusedbatchnormv3_readvariableop_1_resource-batchnorm_4/FusedBatchNormV3:batch_variance:0.^batchnorm_4/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*I
_class?
=;loc:@batchnorm_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batchnorm_4/AssignNewValue_1?
leaky_relu_4/LeakyRelu	LeakyRelu batchnorm_4/FusedBatchNormV3:y:0*/
_output_shapes
:?????????  @*
alpha%???>2
leaky_relu_4/LeakyRelu?
conv_transpose_2/ShapeShape$leaky_relu_4/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv_transpose_2/Shape?
$conv_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_2/strided_slice/stack?
&conv_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_1?
&conv_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_2?
conv_transpose_2/strided_sliceStridedSliceconv_transpose_2/Shape:output:0-conv_transpose_2/strided_slice/stack:output:0/conv_transpose_2/strided_slice/stack_1:output:0/conv_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_2/strided_slicev
conv_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_2/stack/1v
conv_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_2/stack/2v
conv_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_2/stack/3?
conv_transpose_2/stackPack'conv_transpose_2/strided_slice:output:0!conv_transpose_2/stack/1:output:0!conv_transpose_2/stack/2:output:0!conv_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_2/stack?
&conv_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_2/strided_slice_1/stack?
(conv_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_1?
(conv_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_2?
 conv_transpose_2/strided_slice_1StridedSliceconv_transpose_2/stack:output:0/conv_transpose_2/strided_slice_1/stack:output:01conv_transpose_2/strided_slice_1/stack_1:output:01conv_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_2/strided_slice_1?
0conv_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0conv_transpose_2/conv2d_transpose/ReadVariableOp?
!conv_transpose_2/conv2d_transposeConv2DBackpropInputconv_transpose_2/stack:output:08conv_transpose_2/conv2d_transpose/ReadVariableOp:value:0$leaky_relu_4/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2#
!conv_transpose_2/conv2d_transpose?
'conv_transpose_2/BiasAdd/ReadVariableOpReadVariableOp0conv_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'conv_transpose_2/BiasAdd/ReadVariableOp?
conv_transpose_2/BiasAddBiasAdd*conv_transpose_2/conv2d_transpose:output:0/conv_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2
conv_transpose_2/BiasAdd?
batchnorm_5/ReadVariableOpReadVariableOp#batchnorm_5_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm_5/ReadVariableOp?
batchnorm_5/ReadVariableOp_1ReadVariableOp%batchnorm_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm_5/ReadVariableOp_1?
+batchnorm_5/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batchnorm_5/FusedBatchNormV3/ReadVariableOp?
-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_5/FusedBatchNormV3FusedBatchNormV3!conv_transpose_2/BiasAdd:output:0"batchnorm_5/ReadVariableOp:value:0$batchnorm_5/ReadVariableOp_1:value:03batchnorm_5/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
batchnorm_5/FusedBatchNormV3?
batchnorm_5/AssignNewValueAssignVariableOp4batchnorm_5_fusedbatchnormv3_readvariableop_resource)batchnorm_5/FusedBatchNormV3:batch_mean:0,^batchnorm_5/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*G
_class=
;9loc:@batchnorm_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batchnorm_5/AssignNewValue?
batchnorm_5/AssignNewValue_1AssignVariableOp6batchnorm_5_fusedbatchnormv3_readvariableop_1_resource-batchnorm_5/FusedBatchNormV3:batch_variance:0.^batchnorm_5/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*I
_class?
=;loc:@batchnorm_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batchnorm_5/AssignNewValue_1?
leaky_relu_5/LeakyRelu	LeakyRelu batchnorm_5/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@@*
alpha%???>2
leaky_relu_5/LeakyRelu?
conv_transpose_3/ShapeShape$leaky_relu_5/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv_transpose_3/Shape?
$conv_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_3/strided_slice/stack?
&conv_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_1?
&conv_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_2?
conv_transpose_3/strided_sliceStridedSliceconv_transpose_3/Shape:output:0-conv_transpose_3/strided_slice/stack:output:0/conv_transpose_3/strided_slice/stack_1:output:0/conv_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_3/strided_slicew
conv_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_3/stack/1w
conv_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_3/stack/2v
conv_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_3/stack/3?
conv_transpose_3/stackPack'conv_transpose_3/strided_slice:output:0!conv_transpose_3/stack/1:output:0!conv_transpose_3/stack/2:output:0!conv_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_3/stack?
&conv_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_3/strided_slice_1/stack?
(conv_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_1?
(conv_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_2?
 conv_transpose_3/strided_slice_1StridedSliceconv_transpose_3/stack:output:0/conv_transpose_3/strided_slice_1/stack:output:01conv_transpose_3/strided_slice_1/stack_1:output:01conv_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_3/strided_slice_1?
0conv_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype022
0conv_transpose_3/conv2d_transpose/ReadVariableOp?
!conv_transpose_3/conv2d_transposeConv2DBackpropInputconv_transpose_3/stack:output:08conv_transpose_3/conv2d_transpose/ReadVariableOp:value:0$leaky_relu_5/LeakyRelu:activations:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2#
!conv_transpose_3/conv2d_transpose?
'conv_transpose_3/BiasAdd/ReadVariableOpReadVariableOp0conv_transpose_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'conv_transpose_3/BiasAdd/ReadVariableOp?
conv_transpose_3/BiasAddBiasAdd*conv_transpose_3/conv2d_transpose:output:0/conv_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv_transpose_3/BiasAdd?
batchnorm_6/ReadVariableOpReadVariableOp#batchnorm_6_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm_6/ReadVariableOp?
batchnorm_6/ReadVariableOp_1ReadVariableOp%batchnorm_6_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm_6/ReadVariableOp_1?
+batchnorm_6/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02-
+batchnorm_6/FusedBatchNormV3/ReadVariableOp?
-batchnorm_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02/
-batchnorm_6/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_6/FusedBatchNormV3FusedBatchNormV3!conv_transpose_3/BiasAdd:output:0"batchnorm_6/ReadVariableOp:value:0$batchnorm_6/ReadVariableOp_1:value:03batchnorm_6/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
batchnorm_6/FusedBatchNormV3?
batchnorm_6/AssignNewValueAssignVariableOp4batchnorm_6_fusedbatchnormv3_readvariableop_resource)batchnorm_6/FusedBatchNormV3:batch_mean:0,^batchnorm_6/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*G
_class=
;9loc:@batchnorm_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batchnorm_6/AssignNewValue?
batchnorm_6/AssignNewValue_1AssignVariableOp6batchnorm_6_fusedbatchnormv3_readvariableop_1_resource-batchnorm_6/FusedBatchNormV3:batch_variance:0.^batchnorm_6/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*I
_class?
=;loc:@batchnorm_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batchnorm_6/AssignNewValue_1?
leaky_relu_6/LeakyRelu	LeakyRelu batchnorm_6/FusedBatchNormV3:y:0*1
_output_shapes
:??????????? *
alpha%???>2
leaky_relu_6/LeakyRelu?
conv_transpose_4/ShapeShape$leaky_relu_6/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv_transpose_4/Shape?
$conv_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_4/strided_slice/stack?
&conv_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_1?
&conv_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_2?
conv_transpose_4/strided_sliceStridedSliceconv_transpose_4/Shape:output:0-conv_transpose_4/strided_slice/stack:output:0/conv_transpose_4/strided_slice/stack_1:output:0/conv_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_4/strided_slicew
conv_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_4/stack/1w
conv_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_4/stack/2v
conv_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_4/stack/3?
conv_transpose_4/stackPack'conv_transpose_4/strided_slice:output:0!conv_transpose_4/stack/1:output:0!conv_transpose_4/stack/2:output:0!conv_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_4/stack?
&conv_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_4/strided_slice_1/stack?
(conv_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_1?
(conv_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_2?
 conv_transpose_4/strided_slice_1StridedSliceconv_transpose_4/stack:output:0/conv_transpose_4/strided_slice_1/stack:output:01conv_transpose_4/strided_slice_1/stack_1:output:01conv_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_4/strided_slice_1?
0conv_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype022
0conv_transpose_4/conv2d_transpose/ReadVariableOp?
!conv_transpose_4/conv2d_transposeConv2DBackpropInputconv_transpose_4/stack:output:08conv_transpose_4/conv2d_transpose/ReadVariableOp:value:0$leaky_relu_6/LeakyRelu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv_transpose_4/conv2d_transpose?
'conv_transpose_4/BiasAdd/ReadVariableOpReadVariableOp0conv_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv_transpose_4/BiasAdd/ReadVariableOp?
conv_transpose_4/BiasAddBiasAdd*conv_transpose_4/conv2d_transpose:output:0/conv_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv_transpose_4/BiasAdd?
conv_transpose_4/SigmoidSigmoid!conv_transpose_4/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv_transpose_4/Sigmoid?
IdentityIdentityconv_transpose_4/Sigmoid:y:0^batchnorm_1/AssignNewValue^batchnorm_1/AssignNewValue_1,^batchnorm_1/FusedBatchNormV3/ReadVariableOp.^batchnorm_1/FusedBatchNormV3/ReadVariableOp_1^batchnorm_1/ReadVariableOp^batchnorm_1/ReadVariableOp_1^batchnorm_2/AssignNewValue^batchnorm_2/AssignNewValue_1,^batchnorm_2/FusedBatchNormV3/ReadVariableOp.^batchnorm_2/FusedBatchNormV3/ReadVariableOp_1^batchnorm_2/ReadVariableOp^batchnorm_2/ReadVariableOp_1^batchnorm_3/AssignNewValue^batchnorm_3/AssignNewValue_1,^batchnorm_3/FusedBatchNormV3/ReadVariableOp.^batchnorm_3/FusedBatchNormV3/ReadVariableOp_1^batchnorm_3/ReadVariableOp^batchnorm_3/ReadVariableOp_1^batchnorm_4/AssignNewValue^batchnorm_4/AssignNewValue_1,^batchnorm_4/FusedBatchNormV3/ReadVariableOp.^batchnorm_4/FusedBatchNormV3/ReadVariableOp_1^batchnorm_4/ReadVariableOp^batchnorm_4/ReadVariableOp_1^batchnorm_5/AssignNewValue^batchnorm_5/AssignNewValue_1,^batchnorm_5/FusedBatchNormV3/ReadVariableOp.^batchnorm_5/FusedBatchNormV3/ReadVariableOp_1^batchnorm_5/ReadVariableOp^batchnorm_5/ReadVariableOp_1^batchnorm_6/AssignNewValue^batchnorm_6/AssignNewValue_1,^batchnorm_6/FusedBatchNormV3/ReadVariableOp.^batchnorm_6/FusedBatchNormV3/ReadVariableOp_1^batchnorm_6/ReadVariableOp^batchnorm_6/ReadVariableOp_1^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp(^conv_transpose_1/BiasAdd/ReadVariableOp1^conv_transpose_1/conv2d_transpose/ReadVariableOp(^conv_transpose_2/BiasAdd/ReadVariableOp1^conv_transpose_2/conv2d_transpose/ReadVariableOp(^conv_transpose_3/BiasAdd/ReadVariableOp1^conv_transpose_3/conv2d_transpose/ReadVariableOp(^conv_transpose_4/BiasAdd/ReadVariableOp1^conv_transpose_4/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::28
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
batchnorm_3/ReadVariableOp_1batchnorm_3/ReadVariableOp_128
batchnorm_4/AssignNewValuebatchnorm_4/AssignNewValue2<
batchnorm_4/AssignNewValue_1batchnorm_4/AssignNewValue_12Z
+batchnorm_4/FusedBatchNormV3/ReadVariableOp+batchnorm_4/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1-batchnorm_4/FusedBatchNormV3/ReadVariableOp_128
batchnorm_4/ReadVariableOpbatchnorm_4/ReadVariableOp2<
batchnorm_4/ReadVariableOp_1batchnorm_4/ReadVariableOp_128
batchnorm_5/AssignNewValuebatchnorm_5/AssignNewValue2<
batchnorm_5/AssignNewValue_1batchnorm_5/AssignNewValue_12Z
+batchnorm_5/FusedBatchNormV3/ReadVariableOp+batchnorm_5/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1-batchnorm_5/FusedBatchNormV3/ReadVariableOp_128
batchnorm_5/ReadVariableOpbatchnorm_5/ReadVariableOp2<
batchnorm_5/ReadVariableOp_1batchnorm_5/ReadVariableOp_128
batchnorm_6/AssignNewValuebatchnorm_6/AssignNewValue2<
batchnorm_6/AssignNewValue_1batchnorm_6/AssignNewValue_12Z
+batchnorm_6/FusedBatchNormV3/ReadVariableOp+batchnorm_6/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_6/FusedBatchNormV3/ReadVariableOp_1-batchnorm_6/FusedBatchNormV3/ReadVariableOp_128
batchnorm_6/ReadVariableOpbatchnorm_6/ReadVariableOp2<
batchnorm_6/ReadVariableOp_1batchnorm_6/ReadVariableOp_12>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp2R
'conv_transpose_1/BiasAdd/ReadVariableOp'conv_transpose_1/BiasAdd/ReadVariableOp2d
0conv_transpose_1/conv2d_transpose/ReadVariableOp0conv_transpose_1/conv2d_transpose/ReadVariableOp2R
'conv_transpose_2/BiasAdd/ReadVariableOp'conv_transpose_2/BiasAdd/ReadVariableOp2d
0conv_transpose_2/conv2d_transpose/ReadVariableOp0conv_transpose_2/conv2d_transpose/ReadVariableOp2R
'conv_transpose_3/BiasAdd/ReadVariableOp'conv_transpose_3/BiasAdd/ReadVariableOp2d
0conv_transpose_3/conv2d_transpose/ReadVariableOp0conv_transpose_3/conv2d_transpose/ReadVariableOp2R
'conv_transpose_4/BiasAdd/ReadVariableOp'conv_transpose_4/BiasAdd/ReadVariableOp2d
0conv_transpose_4/conv2d_transpose/ReadVariableOp0conv_transpose_4/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_leaky_relu_2_layer_call_and_return_conditional_losses_7126

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@@@*
alpha%???>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_4574

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_6_layer_call_and_return_conditional_losses_7474

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
ͻ
?*
__inference__traced_save_7824
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
6savev2_batchnorm_3_moving_variance_read_readvariableop6
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
6savev2_batchnorm_5_moving_variance_read_readvariableop6
2savev2_conv_transpose_3_kernel_read_readvariableop4
0savev2_conv_transpose_3_bias_read_readvariableop0
,savev2_batchnorm_6_gamma_read_readvariableop/
+savev2_batchnorm_6_beta_read_readvariableop6
2savev2_batchnorm_6_moving_mean_read_readvariableop:
6savev2_batchnorm_6_moving_variance_read_readvariableop6
2savev2_conv_transpose_4_kernel_read_readvariableop4
0savev2_conv_transpose_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv_1_kernel_m_read_readvariableop1
-savev2_adam_conv_1_bias_m_read_readvariableop7
3savev2_adam_batchnorm_1_gamma_m_read_readvariableop6
2savev2_adam_batchnorm_1_beta_m_read_readvariableop3
/savev2_adam_conv_2_kernel_m_read_readvariableop1
-savev2_adam_conv_2_bias_m_read_readvariableop7
3savev2_adam_batchnorm_2_gamma_m_read_readvariableop6
2savev2_adam_batchnorm_2_beta_m_read_readvariableop3
/savev2_adam_conv_3_kernel_m_read_readvariableop1
-savev2_adam_conv_3_bias_m_read_readvariableop7
3savev2_adam_batchnorm_3_gamma_m_read_readvariableop6
2savev2_adam_batchnorm_3_beta_m_read_readvariableop=
9savev2_adam_conv_transpose_1_kernel_m_read_readvariableop;
7savev2_adam_conv_transpose_1_bias_m_read_readvariableop7
3savev2_adam_batchnorm_4_gamma_m_read_readvariableop6
2savev2_adam_batchnorm_4_beta_m_read_readvariableop=
9savev2_adam_conv_transpose_2_kernel_m_read_readvariableop;
7savev2_adam_conv_transpose_2_bias_m_read_readvariableop7
3savev2_adam_batchnorm_5_gamma_m_read_readvariableop6
2savev2_adam_batchnorm_5_beta_m_read_readvariableop=
9savev2_adam_conv_transpose_3_kernel_m_read_readvariableop;
7savev2_adam_conv_transpose_3_bias_m_read_readvariableop7
3savev2_adam_batchnorm_6_gamma_m_read_readvariableop6
2savev2_adam_batchnorm_6_beta_m_read_readvariableop=
9savev2_adam_conv_transpose_4_kernel_m_read_readvariableop;
7savev2_adam_conv_transpose_4_bias_m_read_readvariableop3
/savev2_adam_conv_1_kernel_v_read_readvariableop1
-savev2_adam_conv_1_bias_v_read_readvariableop7
3savev2_adam_batchnorm_1_gamma_v_read_readvariableop6
2savev2_adam_batchnorm_1_beta_v_read_readvariableop3
/savev2_adam_conv_2_kernel_v_read_readvariableop1
-savev2_adam_conv_2_bias_v_read_readvariableop7
3savev2_adam_batchnorm_2_gamma_v_read_readvariableop6
2savev2_adam_batchnorm_2_beta_v_read_readvariableop3
/savev2_adam_conv_3_kernel_v_read_readvariableop1
-savev2_adam_conv_3_bias_v_read_readvariableop7
3savev2_adam_batchnorm_3_gamma_v_read_readvariableop6
2savev2_adam_batchnorm_3_beta_v_read_readvariableop=
9savev2_adam_conv_transpose_1_kernel_v_read_readvariableop;
7savev2_adam_conv_transpose_1_bias_v_read_readvariableop7
3savev2_adam_batchnorm_4_gamma_v_read_readvariableop6
2savev2_adam_batchnorm_4_beta_v_read_readvariableop=
9savev2_adam_conv_transpose_2_kernel_v_read_readvariableop;
7savev2_adam_conv_transpose_2_bias_v_read_readvariableop7
3savev2_adam_batchnorm_5_gamma_v_read_readvariableop6
2savev2_adam_batchnorm_5_beta_v_read_readvariableop=
9savev2_adam_conv_transpose_3_kernel_v_read_readvariableop;
7savev2_adam_conv_transpose_3_bias_v_read_readvariableop7
3savev2_adam_batchnorm_6_gamma_v_read_readvariableop6
2savev2_adam_batchnorm_6_beta_v_read_readvariableop=
9savev2_adam_conv_transpose_4_kernel_v_read_readvariableop;
7savev2_adam_conv_transpose_4_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?6
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:b*
dtype0*?5
value?5B?5bB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:b*
dtype0*?
value?B?bB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?(
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop,savev2_batchnorm_1_gamma_read_readvariableop+savev2_batchnorm_1_beta_read_readvariableop2savev2_batchnorm_1_moving_mean_read_readvariableop6savev2_batchnorm_1_moving_variance_read_readvariableop(savev2_conv_2_kernel_read_readvariableop&savev2_conv_2_bias_read_readvariableop,savev2_batchnorm_2_gamma_read_readvariableop+savev2_batchnorm_2_beta_read_readvariableop2savev2_batchnorm_2_moving_mean_read_readvariableop6savev2_batchnorm_2_moving_variance_read_readvariableop(savev2_conv_3_kernel_read_readvariableop&savev2_conv_3_bias_read_readvariableop,savev2_batchnorm_3_gamma_read_readvariableop+savev2_batchnorm_3_beta_read_readvariableop2savev2_batchnorm_3_moving_mean_read_readvariableop6savev2_batchnorm_3_moving_variance_read_readvariableop2savev2_conv_transpose_1_kernel_read_readvariableop0savev2_conv_transpose_1_bias_read_readvariableop,savev2_batchnorm_4_gamma_read_readvariableop+savev2_batchnorm_4_beta_read_readvariableop2savev2_batchnorm_4_moving_mean_read_readvariableop6savev2_batchnorm_4_moving_variance_read_readvariableop2savev2_conv_transpose_2_kernel_read_readvariableop0savev2_conv_transpose_2_bias_read_readvariableop,savev2_batchnorm_5_gamma_read_readvariableop+savev2_batchnorm_5_beta_read_readvariableop2savev2_batchnorm_5_moving_mean_read_readvariableop6savev2_batchnorm_5_moving_variance_read_readvariableop2savev2_conv_transpose_3_kernel_read_readvariableop0savev2_conv_transpose_3_bias_read_readvariableop,savev2_batchnorm_6_gamma_read_readvariableop+savev2_batchnorm_6_beta_read_readvariableop2savev2_batchnorm_6_moving_mean_read_readvariableop6savev2_batchnorm_6_moving_variance_read_readvariableop2savev2_conv_transpose_4_kernel_read_readvariableop0savev2_conv_transpose_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv_1_kernel_m_read_readvariableop-savev2_adam_conv_1_bias_m_read_readvariableop3savev2_adam_batchnorm_1_gamma_m_read_readvariableop2savev2_adam_batchnorm_1_beta_m_read_readvariableop/savev2_adam_conv_2_kernel_m_read_readvariableop-savev2_adam_conv_2_bias_m_read_readvariableop3savev2_adam_batchnorm_2_gamma_m_read_readvariableop2savev2_adam_batchnorm_2_beta_m_read_readvariableop/savev2_adam_conv_3_kernel_m_read_readvariableop-savev2_adam_conv_3_bias_m_read_readvariableop3savev2_adam_batchnorm_3_gamma_m_read_readvariableop2savev2_adam_batchnorm_3_beta_m_read_readvariableop9savev2_adam_conv_transpose_1_kernel_m_read_readvariableop7savev2_adam_conv_transpose_1_bias_m_read_readvariableop3savev2_adam_batchnorm_4_gamma_m_read_readvariableop2savev2_adam_batchnorm_4_beta_m_read_readvariableop9savev2_adam_conv_transpose_2_kernel_m_read_readvariableop7savev2_adam_conv_transpose_2_bias_m_read_readvariableop3savev2_adam_batchnorm_5_gamma_m_read_readvariableop2savev2_adam_batchnorm_5_beta_m_read_readvariableop9savev2_adam_conv_transpose_3_kernel_m_read_readvariableop7savev2_adam_conv_transpose_3_bias_m_read_readvariableop3savev2_adam_batchnorm_6_gamma_m_read_readvariableop2savev2_adam_batchnorm_6_beta_m_read_readvariableop9savev2_adam_conv_transpose_4_kernel_m_read_readvariableop7savev2_adam_conv_transpose_4_bias_m_read_readvariableop/savev2_adam_conv_1_kernel_v_read_readvariableop-savev2_adam_conv_1_bias_v_read_readvariableop3savev2_adam_batchnorm_1_gamma_v_read_readvariableop2savev2_adam_batchnorm_1_beta_v_read_readvariableop/savev2_adam_conv_2_kernel_v_read_readvariableop-savev2_adam_conv_2_bias_v_read_readvariableop3savev2_adam_batchnorm_2_gamma_v_read_readvariableop2savev2_adam_batchnorm_2_beta_v_read_readvariableop/savev2_adam_conv_3_kernel_v_read_readvariableop-savev2_adam_conv_3_bias_v_read_readvariableop3savev2_adam_batchnorm_3_gamma_v_read_readvariableop2savev2_adam_batchnorm_3_beta_v_read_readvariableop9savev2_adam_conv_transpose_1_kernel_v_read_readvariableop7savev2_adam_conv_transpose_1_bias_v_read_readvariableop3savev2_adam_batchnorm_4_gamma_v_read_readvariableop2savev2_adam_batchnorm_4_beta_v_read_readvariableop9savev2_adam_conv_transpose_2_kernel_v_read_readvariableop7savev2_adam_conv_transpose_2_bias_v_read_readvariableop3savev2_adam_batchnorm_5_gamma_v_read_readvariableop2savev2_adam_batchnorm_5_beta_v_read_readvariableop9savev2_adam_conv_transpose_3_kernel_v_read_readvariableop7savev2_adam_conv_transpose_3_bias_v_read_readvariableop3savev2_adam_batchnorm_6_gamma_v_read_readvariableop2savev2_adam_batchnorm_6_beta_v_read_readvariableop9savev2_adam_conv_transpose_4_kernel_v_read_readvariableop7savev2_adam_conv_transpose_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *p
dtypesf
d2b	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : : @:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@: @: : : : : : :: : : : : : : : : : : : @:@:@:@:@@:@:@:@:@@:@:@:@:@@:@:@:@: @: : : : :: : : : : @:@:@:@:@@:@:@:@:@@:@:@:@:@@:@:@:@: @: : : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
: @:  

_output_shapes
: : !

_output_shapes
: : "

_output_shapes
: : #

_output_shapes
: : $

_output_shapes
: :,%(
&
_output_shapes
: : &

_output_shapes
::'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :,.(
&
_output_shapes
: : /

_output_shapes
: : 0

_output_shapes
: : 1

_output_shapes
: :,2(
&
_output_shapes
: @: 3

_output_shapes
:@: 4

_output_shapes
:@: 5

_output_shapes
:@:,6(
&
_output_shapes
:@@: 7

_output_shapes
:@: 8

_output_shapes
:@: 9

_output_shapes
:@:,:(
&
_output_shapes
:@@: ;

_output_shapes
:@: <

_output_shapes
:@: =

_output_shapes
:@:,>(
&
_output_shapes
:@@: ?

_output_shapes
:@: @

_output_shapes
:@: A

_output_shapes
:@:,B(
&
_output_shapes
: @: C

_output_shapes
: : D

_output_shapes
: : E

_output_shapes
: :,F(
&
_output_shapes
: : G

_output_shapes
::,H(
&
_output_shapes
: : I

_output_shapes
: : J

_output_shapes
: : K

_output_shapes
: :,L(
&
_output_shapes
: @: M

_output_shapes
:@: N

_output_shapes
:@: O

_output_shapes
:@:,P(
&
_output_shapes
:@@: Q

_output_shapes
:@: R

_output_shapes
:@: S

_output_shapes
:@:,T(
&
_output_shapes
:@@: U

_output_shapes
:@: V

_output_shapes
:@: W

_output_shapes
:@:,X(
&
_output_shapes
:@@: Y

_output_shapes
:@: Z

_output_shapes
:@: [

_output_shapes
:@:,\(
&
_output_shapes
: @: ]

_output_shapes
: : ^

_output_shapes
: : _

_output_shapes
: :,`(
&
_output_shapes
: : a

_output_shapes
::b

_output_shapes
: 
?
b
F__inference_leaky_relu_5_layer_call_and_return_conditional_losses_7431

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+???????????????????????????@*
alpha%???>2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?#
?
J__inference_conv_transpose_1_layer_call_and_return_conditional_losses_4754

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
b
F__inference_leaky_relu_3_layer_call_and_return_conditional_losses_5541

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????  @*
alpha%???>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
G
+__inference_leaky_relu_5_layer_call_fn_7436

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_5_layer_call_and_return_conditional_losses_56472
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?4
 __inference__traced_restore_8125
file_prefix"
assignvariableop_conv_1_kernel"
assignvariableop_1_conv_1_bias(
$assignvariableop_2_batchnorm_1_gamma'
#assignvariableop_3_batchnorm_1_beta.
*assignvariableop_4_batchnorm_1_moving_mean2
.assignvariableop_5_batchnorm_1_moving_variance$
 assignvariableop_6_conv_2_kernel"
assignvariableop_7_conv_2_bias(
$assignvariableop_8_batchnorm_2_gamma'
#assignvariableop_9_batchnorm_2_beta/
+assignvariableop_10_batchnorm_2_moving_mean3
/assignvariableop_11_batchnorm_2_moving_variance%
!assignvariableop_12_conv_3_kernel#
assignvariableop_13_conv_3_bias)
%assignvariableop_14_batchnorm_3_gamma(
$assignvariableop_15_batchnorm_3_beta/
+assignvariableop_16_batchnorm_3_moving_mean3
/assignvariableop_17_batchnorm_3_moving_variance/
+assignvariableop_18_conv_transpose_1_kernel-
)assignvariableop_19_conv_transpose_1_bias)
%assignvariableop_20_batchnorm_4_gamma(
$assignvariableop_21_batchnorm_4_beta/
+assignvariableop_22_batchnorm_4_moving_mean3
/assignvariableop_23_batchnorm_4_moving_variance/
+assignvariableop_24_conv_transpose_2_kernel-
)assignvariableop_25_conv_transpose_2_bias)
%assignvariableop_26_batchnorm_5_gamma(
$assignvariableop_27_batchnorm_5_beta/
+assignvariableop_28_batchnorm_5_moving_mean3
/assignvariableop_29_batchnorm_5_moving_variance/
+assignvariableop_30_conv_transpose_3_kernel-
)assignvariableop_31_conv_transpose_3_bias)
%assignvariableop_32_batchnorm_6_gamma(
$assignvariableop_33_batchnorm_6_beta/
+assignvariableop_34_batchnorm_6_moving_mean3
/assignvariableop_35_batchnorm_6_moving_variance/
+assignvariableop_36_conv_transpose_4_kernel-
)assignvariableop_37_conv_transpose_4_bias!
assignvariableop_38_adam_iter#
assignvariableop_39_adam_beta_1#
assignvariableop_40_adam_beta_2"
assignvariableop_41_adam_decay*
&assignvariableop_42_adam_learning_rate
assignvariableop_43_total
assignvariableop_44_count,
(assignvariableop_45_adam_conv_1_kernel_m*
&assignvariableop_46_adam_conv_1_bias_m0
,assignvariableop_47_adam_batchnorm_1_gamma_m/
+assignvariableop_48_adam_batchnorm_1_beta_m,
(assignvariableop_49_adam_conv_2_kernel_m*
&assignvariableop_50_adam_conv_2_bias_m0
,assignvariableop_51_adam_batchnorm_2_gamma_m/
+assignvariableop_52_adam_batchnorm_2_beta_m,
(assignvariableop_53_adam_conv_3_kernel_m*
&assignvariableop_54_adam_conv_3_bias_m0
,assignvariableop_55_adam_batchnorm_3_gamma_m/
+assignvariableop_56_adam_batchnorm_3_beta_m6
2assignvariableop_57_adam_conv_transpose_1_kernel_m4
0assignvariableop_58_adam_conv_transpose_1_bias_m0
,assignvariableop_59_adam_batchnorm_4_gamma_m/
+assignvariableop_60_adam_batchnorm_4_beta_m6
2assignvariableop_61_adam_conv_transpose_2_kernel_m4
0assignvariableop_62_adam_conv_transpose_2_bias_m0
,assignvariableop_63_adam_batchnorm_5_gamma_m/
+assignvariableop_64_adam_batchnorm_5_beta_m6
2assignvariableop_65_adam_conv_transpose_3_kernel_m4
0assignvariableop_66_adam_conv_transpose_3_bias_m0
,assignvariableop_67_adam_batchnorm_6_gamma_m/
+assignvariableop_68_adam_batchnorm_6_beta_m6
2assignvariableop_69_adam_conv_transpose_4_kernel_m4
0assignvariableop_70_adam_conv_transpose_4_bias_m,
(assignvariableop_71_adam_conv_1_kernel_v*
&assignvariableop_72_adam_conv_1_bias_v0
,assignvariableop_73_adam_batchnorm_1_gamma_v/
+assignvariableop_74_adam_batchnorm_1_beta_v,
(assignvariableop_75_adam_conv_2_kernel_v*
&assignvariableop_76_adam_conv_2_bias_v0
,assignvariableop_77_adam_batchnorm_2_gamma_v/
+assignvariableop_78_adam_batchnorm_2_beta_v,
(assignvariableop_79_adam_conv_3_kernel_v*
&assignvariableop_80_adam_conv_3_bias_v0
,assignvariableop_81_adam_batchnorm_3_gamma_v/
+assignvariableop_82_adam_batchnorm_3_beta_v6
2assignvariableop_83_adam_conv_transpose_1_kernel_v4
0assignvariableop_84_adam_conv_transpose_1_bias_v0
,assignvariableop_85_adam_batchnorm_4_gamma_v/
+assignvariableop_86_adam_batchnorm_4_beta_v6
2assignvariableop_87_adam_conv_transpose_2_kernel_v4
0assignvariableop_88_adam_conv_transpose_2_bias_v0
,assignvariableop_89_adam_batchnorm_5_gamma_v/
+assignvariableop_90_adam_batchnorm_5_beta_v6
2assignvariableop_91_adam_conv_transpose_3_kernel_v4
0assignvariableop_92_adam_conv_transpose_3_bias_v0
,assignvariableop_93_adam_batchnorm_6_gamma_v/
+assignvariableop_94_adam_batchnorm_6_beta_v6
2assignvariableop_95_adam_conv_transpose_4_kernel_v4
0assignvariableop_96_adam_conv_transpose_4_bias_v
identity_98??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?6
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:b*
dtype0*?5
value?5B?5bB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:b*
dtype0*?
value?B?bB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*p
dtypesf
d2b	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp$assignvariableop_2_batchnorm_1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp#assignvariableop_3_batchnorm_1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp*assignvariableop_4_batchnorm_1_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batchnorm_1_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp$assignvariableop_8_batchnorm_2_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_batchnorm_2_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp+assignvariableop_10_batchnorm_2_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batchnorm_2_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_conv_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp%assignvariableop_14_batchnorm_3_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp$assignvariableop_15_batchnorm_3_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp+assignvariableop_16_batchnorm_3_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batchnorm_3_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp+assignvariableop_18_conv_transpose_1_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp)assignvariableop_19_conv_transpose_1_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp%assignvariableop_20_batchnorm_4_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp$assignvariableop_21_batchnorm_4_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp+assignvariableop_22_batchnorm_4_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp/assignvariableop_23_batchnorm_4_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp+assignvariableop_24_conv_transpose_2_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp)assignvariableop_25_conv_transpose_2_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp%assignvariableop_26_batchnorm_5_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp$assignvariableop_27_batchnorm_5_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_batchnorm_5_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp/assignvariableop_29_batchnorm_5_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp+assignvariableop_30_conv_transpose_3_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_conv_transpose_3_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp%assignvariableop_32_batchnorm_6_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp$assignvariableop_33_batchnorm_6_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp+assignvariableop_34_batchnorm_6_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp/assignvariableop_35_batchnorm_6_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp+assignvariableop_36_conv_transpose_4_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp)assignvariableop_37_conv_transpose_4_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpassignvariableop_38_adam_iterIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpassignvariableop_39_adam_beta_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpassignvariableop_40_adam_beta_2Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_adam_decayIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_learning_rateIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpassignvariableop_43_totalIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpassignvariableop_44_countIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_conv_1_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp&assignvariableop_46_adam_conv_1_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp,assignvariableop_47_adam_batchnorm_1_gamma_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp+assignvariableop_48_adam_batchnorm_1_beta_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_conv_2_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp&assignvariableop_50_adam_conv_2_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp,assignvariableop_51_adam_batchnorm_2_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp+assignvariableop_52_adam_batchnorm_2_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_conv_3_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp&assignvariableop_54_adam_conv_3_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp,assignvariableop_55_adam_batchnorm_3_gamma_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp+assignvariableop_56_adam_batchnorm_3_beta_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp2assignvariableop_57_adam_conv_transpose_1_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp0assignvariableop_58_adam_conv_transpose_1_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp,assignvariableop_59_adam_batchnorm_4_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp+assignvariableop_60_adam_batchnorm_4_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp2assignvariableop_61_adam_conv_transpose_2_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp0assignvariableop_62_adam_conv_transpose_2_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp,assignvariableop_63_adam_batchnorm_5_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp+assignvariableop_64_adam_batchnorm_5_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp2assignvariableop_65_adam_conv_transpose_3_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp0assignvariableop_66_adam_conv_transpose_3_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp,assignvariableop_67_adam_batchnorm_6_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp+assignvariableop_68_adam_batchnorm_6_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp2assignvariableop_69_adam_conv_transpose_4_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp0assignvariableop_70_adam_conv_transpose_4_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp(assignvariableop_71_adam_conv_1_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp&assignvariableop_72_adam_conv_1_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp,assignvariableop_73_adam_batchnorm_1_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp+assignvariableop_74_adam_batchnorm_1_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp(assignvariableop_75_adam_conv_2_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp&assignvariableop_76_adam_conv_2_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp,assignvariableop_77_adam_batchnorm_2_gamma_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp+assignvariableop_78_adam_batchnorm_2_beta_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp(assignvariableop_79_adam_conv_3_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp&assignvariableop_80_adam_conv_3_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp,assignvariableop_81_adam_batchnorm_3_gamma_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp+assignvariableop_82_adam_batchnorm_3_beta_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp2assignvariableop_83_adam_conv_transpose_1_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp0assignvariableop_84_adam_conv_transpose_1_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp,assignvariableop_85_adam_batchnorm_4_gamma_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp+assignvariableop_86_adam_batchnorm_4_beta_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp2assignvariableop_87_adam_conv_transpose_2_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp0assignvariableop_88_adam_conv_transpose_2_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp,assignvariableop_89_adam_batchnorm_5_gamma_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp+assignvariableop_90_adam_batchnorm_5_beta_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp2assignvariableop_91_adam_conv_transpose_3_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp0assignvariableop_92_adam_conv_transpose_3_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp,assignvariableop_93_adam_batchnorm_6_gamma_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp+assignvariableop_94_adam_batchnorm_6_beta_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp2assignvariableop_95_adam_conv_transpose_4_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp0assignvariableop_96_adam_conv_transpose_4_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_969
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_97Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_97?
Identity_98IdentityIdentity_97:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96*
T0*
_output_shapes
: 2
Identity_98"#
identity_98Identity_98:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
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
AssignVariableOp_96AssignVariableOp_96:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_6938

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
z
%__inference_conv_1_layer_call_fn_6836

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv_1_layer_call_and_return_conditional_losses_52232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?#
?
J__inference_conv_transpose_3_layer_call_and_return_conditional_losses_5050

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_6874

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_7308

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_4501

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_6920

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
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
?
b
F__inference_leaky_relu_5_layer_call_and_return_conditional_losses_5647

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+???????????????????????????@*
alpha%???>2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
*__inference_batchnorm_2_layer_call_fn_7121

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_53882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@@@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
G
+__inference_leaky_relu_6_layer_call_fn_7510

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_6_layer_call_and_return_conditional_losses_57002
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_5500

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_6655

inputs)
%conv_1_conv2d_readvariableop_resource*
&conv_1_biasadd_readvariableop_resource'
#batchnorm_1_readvariableop_resource)
%batchnorm_1_readvariableop_1_resource8
4batchnorm_1_fusedbatchnormv3_readvariableop_resource:
6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource)
%conv_2_conv2d_readvariableop_resource*
&conv_2_biasadd_readvariableop_resource'
#batchnorm_2_readvariableop_resource)
%batchnorm_2_readvariableop_1_resource8
4batchnorm_2_fusedbatchnormv3_readvariableop_resource:
6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource)
%conv_3_conv2d_readvariableop_resource*
&conv_3_biasadd_readvariableop_resource'
#batchnorm_3_readvariableop_resource)
%batchnorm_3_readvariableop_1_resource8
4batchnorm_3_fusedbatchnormv3_readvariableop_resource:
6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource=
9conv_transpose_1_conv2d_transpose_readvariableop_resource4
0conv_transpose_1_biasadd_readvariableop_resource'
#batchnorm_4_readvariableop_resource)
%batchnorm_4_readvariableop_1_resource8
4batchnorm_4_fusedbatchnormv3_readvariableop_resource:
6batchnorm_4_fusedbatchnormv3_readvariableop_1_resource=
9conv_transpose_2_conv2d_transpose_readvariableop_resource4
0conv_transpose_2_biasadd_readvariableop_resource'
#batchnorm_5_readvariableop_resource)
%batchnorm_5_readvariableop_1_resource8
4batchnorm_5_fusedbatchnormv3_readvariableop_resource:
6batchnorm_5_fusedbatchnormv3_readvariableop_1_resource=
9conv_transpose_3_conv2d_transpose_readvariableop_resource4
0conv_transpose_3_biasadd_readvariableop_resource'
#batchnorm_6_readvariableop_resource)
%batchnorm_6_readvariableop_1_resource8
4batchnorm_6_fusedbatchnormv3_readvariableop_resource:
6batchnorm_6_fusedbatchnormv3_readvariableop_1_resource=
9conv_transpose_4_conv2d_transpose_readvariableop_resource4
0conv_transpose_4_biasadd_readvariableop_resource
identity??+batchnorm_1/FusedBatchNormV3/ReadVariableOp?-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?batchnorm_1/ReadVariableOp?batchnorm_1/ReadVariableOp_1?+batchnorm_2/FusedBatchNormV3/ReadVariableOp?-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1?batchnorm_2/ReadVariableOp?batchnorm_2/ReadVariableOp_1?+batchnorm_3/FusedBatchNormV3/ReadVariableOp?-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1?batchnorm_3/ReadVariableOp?batchnorm_3/ReadVariableOp_1?+batchnorm_4/FusedBatchNormV3/ReadVariableOp?-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1?batchnorm_4/ReadVariableOp?batchnorm_4/ReadVariableOp_1?+batchnorm_5/FusedBatchNormV3/ReadVariableOp?-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1?batchnorm_5/ReadVariableOp?batchnorm_5/ReadVariableOp_1?+batchnorm_6/FusedBatchNormV3/ReadVariableOp?-batchnorm_6/FusedBatchNormV3/ReadVariableOp_1?batchnorm_6/ReadVariableOp?batchnorm_6/ReadVariableOp_1?conv_1/BiasAdd/ReadVariableOp?conv_1/Conv2D/ReadVariableOp?conv_2/BiasAdd/ReadVariableOp?conv_2/Conv2D/ReadVariableOp?conv_3/BiasAdd/ReadVariableOp?conv_3/Conv2D/ReadVariableOp?'conv_transpose_1/BiasAdd/ReadVariableOp?0conv_transpose_1/conv2d_transpose/ReadVariableOp?'conv_transpose_2/BiasAdd/ReadVariableOp?0conv_transpose_2/conv2d_transpose/ReadVariableOp?'conv_transpose_3/BiasAdd/ReadVariableOp?0conv_transpose_3/conv2d_transpose/ReadVariableOp?'conv_transpose_4/BiasAdd/ReadVariableOp?0conv_transpose_4/conv2d_transpose/ReadVariableOp?
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv_1/Conv2D/ReadVariableOp?
conv_1/Conv2DConv2Dinputs$conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv_1/Conv2D?
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv_1/BiasAdd/ReadVariableOp?
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv_1/BiasAdd?
batchnorm_1/ReadVariableOpReadVariableOp#batchnorm_1_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm_1/ReadVariableOp?
batchnorm_1/ReadVariableOp_1ReadVariableOp%batchnorm_1_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm_1/ReadVariableOp_1?
+batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02-
+batchnorm_1/FusedBatchNormV3/ReadVariableOp?
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02/
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_1/FusedBatchNormV3FusedBatchNormV3conv_1/BiasAdd:output:0"batchnorm_1/ReadVariableOp:value:0$batchnorm_1/ReadVariableOp_1:value:03batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
batchnorm_1/FusedBatchNormV3?
leaky_relu_1/LeakyRelu	LeakyRelu batchnorm_1/FusedBatchNormV3:y:0*1
_output_shapes
:??????????? *
alpha%???>2
leaky_relu_1/LeakyRelu?
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv_2/Conv2D/ReadVariableOp?
conv_2/Conv2DConv2D$leaky_relu_1/LeakyRelu:activations:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
conv_2/Conv2D?
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_2/BiasAdd/ReadVariableOp?
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2
conv_2/BiasAdd?
batchnorm_2/ReadVariableOpReadVariableOp#batchnorm_2_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm_2/ReadVariableOp?
batchnorm_2/ReadVariableOp_1ReadVariableOp%batchnorm_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm_2/ReadVariableOp_1?
+batchnorm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batchnorm_2/FusedBatchNormV3/ReadVariableOp?
-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_2/FusedBatchNormV3FusedBatchNormV3conv_2/BiasAdd:output:0"batchnorm_2/ReadVariableOp:value:0$batchnorm_2/ReadVariableOp_1:value:03batchnorm_2/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( 2
batchnorm_2/FusedBatchNormV3?
leaky_relu_2/LeakyRelu	LeakyRelu batchnorm_2/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@@*
alpha%???>2
leaky_relu_2/LeakyRelu?
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv_3/Conv2D/ReadVariableOp?
conv_3/Conv2DConv2D$leaky_relu_2/LeakyRelu:activations:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
conv_3/Conv2D?
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_3/BiasAdd/ReadVariableOp?
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
conv_3/BiasAdd?
batchnorm_3/ReadVariableOpReadVariableOp#batchnorm_3_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm_3/ReadVariableOp?
batchnorm_3/ReadVariableOp_1ReadVariableOp%batchnorm_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm_3/ReadVariableOp_1?
+batchnorm_3/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batchnorm_3/FusedBatchNormV3/ReadVariableOp?
-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_3/FusedBatchNormV3FusedBatchNormV3conv_3/BiasAdd:output:0"batchnorm_3/ReadVariableOp:value:0$batchnorm_3/ReadVariableOp_1:value:03batchnorm_3/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2
batchnorm_3/FusedBatchNormV3?
leaky_relu_3/LeakyRelu	LeakyRelu batchnorm_3/FusedBatchNormV3:y:0*/
_output_shapes
:?????????  @*
alpha%???>2
leaky_relu_3/LeakyRelu?
conv_transpose_1/ShapeShape$leaky_relu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv_transpose_1/Shape?
$conv_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_1/strided_slice/stack?
&conv_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_1?
&conv_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_2?
conv_transpose_1/strided_sliceStridedSliceconv_transpose_1/Shape:output:0-conv_transpose_1/strided_slice/stack:output:0/conv_transpose_1/strided_slice/stack_1:output:0/conv_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_1/strided_slicev
conv_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_1/stack/1v
conv_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_1/stack/2v
conv_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_1/stack/3?
conv_transpose_1/stackPack'conv_transpose_1/strided_slice:output:0!conv_transpose_1/stack/1:output:0!conv_transpose_1/stack/2:output:0!conv_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_1/stack?
&conv_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_1/strided_slice_1/stack?
(conv_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_1?
(conv_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_2?
 conv_transpose_1/strided_slice_1StridedSliceconv_transpose_1/stack:output:0/conv_transpose_1/strided_slice_1/stack:output:01conv_transpose_1/strided_slice_1/stack_1:output:01conv_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_1/strided_slice_1?
0conv_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0conv_transpose_1/conv2d_transpose/ReadVariableOp?
!conv_transpose_1/conv2d_transposeConv2DBackpropInputconv_transpose_1/stack:output:08conv_transpose_1/conv2d_transpose/ReadVariableOp:value:0$leaky_relu_3/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2#
!conv_transpose_1/conv2d_transpose?
'conv_transpose_1/BiasAdd/ReadVariableOpReadVariableOp0conv_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'conv_transpose_1/BiasAdd/ReadVariableOp?
conv_transpose_1/BiasAddBiasAdd*conv_transpose_1/conv2d_transpose:output:0/conv_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
conv_transpose_1/BiasAdd?
batchnorm_4/ReadVariableOpReadVariableOp#batchnorm_4_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm_4/ReadVariableOp?
batchnorm_4/ReadVariableOp_1ReadVariableOp%batchnorm_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm_4/ReadVariableOp_1?
+batchnorm_4/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batchnorm_4/FusedBatchNormV3/ReadVariableOp?
-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_4/FusedBatchNormV3FusedBatchNormV3!conv_transpose_1/BiasAdd:output:0"batchnorm_4/ReadVariableOp:value:0$batchnorm_4/ReadVariableOp_1:value:03batchnorm_4/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2
batchnorm_4/FusedBatchNormV3?
leaky_relu_4/LeakyRelu	LeakyRelu batchnorm_4/FusedBatchNormV3:y:0*/
_output_shapes
:?????????  @*
alpha%???>2
leaky_relu_4/LeakyRelu?
conv_transpose_2/ShapeShape$leaky_relu_4/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv_transpose_2/Shape?
$conv_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_2/strided_slice/stack?
&conv_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_1?
&conv_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_2?
conv_transpose_2/strided_sliceStridedSliceconv_transpose_2/Shape:output:0-conv_transpose_2/strided_slice/stack:output:0/conv_transpose_2/strided_slice/stack_1:output:0/conv_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_2/strided_slicev
conv_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_2/stack/1v
conv_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_2/stack/2v
conv_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_2/stack/3?
conv_transpose_2/stackPack'conv_transpose_2/strided_slice:output:0!conv_transpose_2/stack/1:output:0!conv_transpose_2/stack/2:output:0!conv_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_2/stack?
&conv_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_2/strided_slice_1/stack?
(conv_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_1?
(conv_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_2?
 conv_transpose_2/strided_slice_1StridedSliceconv_transpose_2/stack:output:0/conv_transpose_2/strided_slice_1/stack:output:01conv_transpose_2/strided_slice_1/stack_1:output:01conv_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_2/strided_slice_1?
0conv_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0conv_transpose_2/conv2d_transpose/ReadVariableOp?
!conv_transpose_2/conv2d_transposeConv2DBackpropInputconv_transpose_2/stack:output:08conv_transpose_2/conv2d_transpose/ReadVariableOp:value:0$leaky_relu_4/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2#
!conv_transpose_2/conv2d_transpose?
'conv_transpose_2/BiasAdd/ReadVariableOpReadVariableOp0conv_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'conv_transpose_2/BiasAdd/ReadVariableOp?
conv_transpose_2/BiasAddBiasAdd*conv_transpose_2/conv2d_transpose:output:0/conv_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2
conv_transpose_2/BiasAdd?
batchnorm_5/ReadVariableOpReadVariableOp#batchnorm_5_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm_5/ReadVariableOp?
batchnorm_5/ReadVariableOp_1ReadVariableOp%batchnorm_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm_5/ReadVariableOp_1?
+batchnorm_5/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batchnorm_5/FusedBatchNormV3/ReadVariableOp?
-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_5/FusedBatchNormV3FusedBatchNormV3!conv_transpose_2/BiasAdd:output:0"batchnorm_5/ReadVariableOp:value:0$batchnorm_5/ReadVariableOp_1:value:03batchnorm_5/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( 2
batchnorm_5/FusedBatchNormV3?
leaky_relu_5/LeakyRelu	LeakyRelu batchnorm_5/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@@*
alpha%???>2
leaky_relu_5/LeakyRelu?
conv_transpose_3/ShapeShape$leaky_relu_5/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv_transpose_3/Shape?
$conv_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_3/strided_slice/stack?
&conv_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_1?
&conv_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_2?
conv_transpose_3/strided_sliceStridedSliceconv_transpose_3/Shape:output:0-conv_transpose_3/strided_slice/stack:output:0/conv_transpose_3/strided_slice/stack_1:output:0/conv_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_3/strided_slicew
conv_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_3/stack/1w
conv_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_3/stack/2v
conv_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_3/stack/3?
conv_transpose_3/stackPack'conv_transpose_3/strided_slice:output:0!conv_transpose_3/stack/1:output:0!conv_transpose_3/stack/2:output:0!conv_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_3/stack?
&conv_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_3/strided_slice_1/stack?
(conv_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_1?
(conv_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_2?
 conv_transpose_3/strided_slice_1StridedSliceconv_transpose_3/stack:output:0/conv_transpose_3/strided_slice_1/stack:output:01conv_transpose_3/strided_slice_1/stack_1:output:01conv_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_3/strided_slice_1?
0conv_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype022
0conv_transpose_3/conv2d_transpose/ReadVariableOp?
!conv_transpose_3/conv2d_transposeConv2DBackpropInputconv_transpose_3/stack:output:08conv_transpose_3/conv2d_transpose/ReadVariableOp:value:0$leaky_relu_5/LeakyRelu:activations:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2#
!conv_transpose_3/conv2d_transpose?
'conv_transpose_3/BiasAdd/ReadVariableOpReadVariableOp0conv_transpose_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'conv_transpose_3/BiasAdd/ReadVariableOp?
conv_transpose_3/BiasAddBiasAdd*conv_transpose_3/conv2d_transpose:output:0/conv_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv_transpose_3/BiasAdd?
batchnorm_6/ReadVariableOpReadVariableOp#batchnorm_6_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm_6/ReadVariableOp?
batchnorm_6/ReadVariableOp_1ReadVariableOp%batchnorm_6_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm_6/ReadVariableOp_1?
+batchnorm_6/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02-
+batchnorm_6/FusedBatchNormV3/ReadVariableOp?
-batchnorm_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02/
-batchnorm_6/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_6/FusedBatchNormV3FusedBatchNormV3!conv_transpose_3/BiasAdd:output:0"batchnorm_6/ReadVariableOp:value:0$batchnorm_6/ReadVariableOp_1:value:03batchnorm_6/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
batchnorm_6/FusedBatchNormV3?
leaky_relu_6/LeakyRelu	LeakyRelu batchnorm_6/FusedBatchNormV3:y:0*1
_output_shapes
:??????????? *
alpha%???>2
leaky_relu_6/LeakyRelu?
conv_transpose_4/ShapeShape$leaky_relu_6/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv_transpose_4/Shape?
$conv_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_4/strided_slice/stack?
&conv_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_1?
&conv_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_2?
conv_transpose_4/strided_sliceStridedSliceconv_transpose_4/Shape:output:0-conv_transpose_4/strided_slice/stack:output:0/conv_transpose_4/strided_slice/stack_1:output:0/conv_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_4/strided_slicew
conv_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_4/stack/1w
conv_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_4/stack/2v
conv_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_4/stack/3?
conv_transpose_4/stackPack'conv_transpose_4/strided_slice:output:0!conv_transpose_4/stack/1:output:0!conv_transpose_4/stack/2:output:0!conv_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_4/stack?
&conv_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_4/strided_slice_1/stack?
(conv_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_1?
(conv_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_2?
 conv_transpose_4/strided_slice_1StridedSliceconv_transpose_4/stack:output:0/conv_transpose_4/strided_slice_1/stack:output:01conv_transpose_4/strided_slice_1/stack_1:output:01conv_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_4/strided_slice_1?
0conv_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype022
0conv_transpose_4/conv2d_transpose/ReadVariableOp?
!conv_transpose_4/conv2d_transposeConv2DBackpropInputconv_transpose_4/stack:output:08conv_transpose_4/conv2d_transpose/ReadVariableOp:value:0$leaky_relu_6/LeakyRelu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv_transpose_4/conv2d_transpose?
'conv_transpose_4/BiasAdd/ReadVariableOpReadVariableOp0conv_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'conv_transpose_4/BiasAdd/ReadVariableOp?
conv_transpose_4/BiasAddBiasAdd*conv_transpose_4/conv2d_transpose:output:0/conv_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv_transpose_4/BiasAdd?
conv_transpose_4/SigmoidSigmoid!conv_transpose_4/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv_transpose_4/Sigmoid?
IdentityIdentityconv_transpose_4/Sigmoid:y:0,^batchnorm_1/FusedBatchNormV3/ReadVariableOp.^batchnorm_1/FusedBatchNormV3/ReadVariableOp_1^batchnorm_1/ReadVariableOp^batchnorm_1/ReadVariableOp_1,^batchnorm_2/FusedBatchNormV3/ReadVariableOp.^batchnorm_2/FusedBatchNormV3/ReadVariableOp_1^batchnorm_2/ReadVariableOp^batchnorm_2/ReadVariableOp_1,^batchnorm_3/FusedBatchNormV3/ReadVariableOp.^batchnorm_3/FusedBatchNormV3/ReadVariableOp_1^batchnorm_3/ReadVariableOp^batchnorm_3/ReadVariableOp_1,^batchnorm_4/FusedBatchNormV3/ReadVariableOp.^batchnorm_4/FusedBatchNormV3/ReadVariableOp_1^batchnorm_4/ReadVariableOp^batchnorm_4/ReadVariableOp_1,^batchnorm_5/FusedBatchNormV3/ReadVariableOp.^batchnorm_5/FusedBatchNormV3/ReadVariableOp_1^batchnorm_5/ReadVariableOp^batchnorm_5/ReadVariableOp_1,^batchnorm_6/FusedBatchNormV3/ReadVariableOp.^batchnorm_6/FusedBatchNormV3/ReadVariableOp_1^batchnorm_6/ReadVariableOp^batchnorm_6/ReadVariableOp_1^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp(^conv_transpose_1/BiasAdd/ReadVariableOp1^conv_transpose_1/conv2d_transpose/ReadVariableOp(^conv_transpose_2/BiasAdd/ReadVariableOp1^conv_transpose_2/conv2d_transpose/ReadVariableOp(^conv_transpose_3/BiasAdd/ReadVariableOp1^conv_transpose_3/conv2d_transpose/ReadVariableOp(^conv_transpose_4/BiasAdd/ReadVariableOp1^conv_transpose_4/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::2Z
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
batchnorm_3/ReadVariableOp_1batchnorm_3/ReadVariableOp_12Z
+batchnorm_4/FusedBatchNormV3/ReadVariableOp+batchnorm_4/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1-batchnorm_4/FusedBatchNormV3/ReadVariableOp_128
batchnorm_4/ReadVariableOpbatchnorm_4/ReadVariableOp2<
batchnorm_4/ReadVariableOp_1batchnorm_4/ReadVariableOp_12Z
+batchnorm_5/FusedBatchNormV3/ReadVariableOp+batchnorm_5/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1-batchnorm_5/FusedBatchNormV3/ReadVariableOp_128
batchnorm_5/ReadVariableOpbatchnorm_5/ReadVariableOp2<
batchnorm_5/ReadVariableOp_1batchnorm_5/ReadVariableOp_12Z
+batchnorm_6/FusedBatchNormV3/ReadVariableOp+batchnorm_6/FusedBatchNormV3/ReadVariableOp2^
-batchnorm_6/FusedBatchNormV3/ReadVariableOp_1-batchnorm_6/FusedBatchNormV3/ReadVariableOp_128
batchnorm_6/ReadVariableOpbatchnorm_6/ReadVariableOp2<
batchnorm_6/ReadVariableOp_1batchnorm_6/ReadVariableOp_12>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp2R
'conv_transpose_1/BiasAdd/ReadVariableOp'conv_transpose_1/BiasAdd/ReadVariableOp2d
0conv_transpose_1/conv2d_transpose/ReadVariableOp0conv_transpose_1/conv2d_transpose/ReadVariableOp2R
'conv_transpose_2/BiasAdd/ReadVariableOp'conv_transpose_2/BiasAdd/ReadVariableOp2d
0conv_transpose_2/conv2d_transpose/ReadVariableOp0conv_transpose_2/conv2d_transpose/ReadVariableOp2R
'conv_transpose_3/BiasAdd/ReadVariableOp'conv_transpose_3/BiasAdd/ReadVariableOp2d
0conv_transpose_3/conv2d_transpose/ReadVariableOp0conv_transpose_3/conv2d_transpose/ReadVariableOp2R
'conv_transpose_4/BiasAdd/ReadVariableOp'conv_transpose_4/BiasAdd/ReadVariableOp2d
0conv_transpose_4/conv2d_transpose/ReadVariableOp0conv_transpose_4/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
G
+__inference_leaky_relu_1_layer_call_fn_6974

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
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_leaky_relu_1_layer_call_and_return_conditional_losses_53172
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs"?L
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
serving_default_input_layer:0???????????N
conv_transpose_4:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
??
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
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_networkʮ{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batchnorm_1", "inbound_nodes": [[["conv_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_relu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_relu_1", "inbound_nodes": [[["batchnorm_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["leaky_relu_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batchnorm_2", "inbound_nodes": [[["conv_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_relu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_relu_2", "inbound_nodes": [[["batchnorm_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["leaky_relu_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batchnorm_3", "inbound_nodes": [[["conv_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_relu_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_relu_3", "inbound_nodes": [[["batchnorm_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv_transpose_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv_transpose_1", "inbound_nodes": [[["leaky_relu_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batchnorm_4", "inbound_nodes": [[["conv_transpose_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_relu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_relu_4", "inbound_nodes": [[["batchnorm_4", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv_transpose_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv_transpose_2", "inbound_nodes": [[["leaky_relu_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batchnorm_5", "inbound_nodes": [[["conv_transpose_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_relu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_relu_5", "inbound_nodes": [[["batchnorm_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv_transpose_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv_transpose_3", "inbound_nodes": [[["leaky_relu_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batchnorm_6", "inbound_nodes": [[["conv_transpose_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_relu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_relu_6", "inbound_nodes": [[["batchnorm_6", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv_transpose_4", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv_transpose_4", "inbound_nodes": [[["leaky_relu_6", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["conv_transpose_4", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batchnorm_1", "inbound_nodes": [[["conv_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_relu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_relu_1", "inbound_nodes": [[["batchnorm_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["leaky_relu_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batchnorm_2", "inbound_nodes": [[["conv_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_relu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_relu_2", "inbound_nodes": [[["batchnorm_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["leaky_relu_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batchnorm_3", "inbound_nodes": [[["conv_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_relu_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_relu_3", "inbound_nodes": [[["batchnorm_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv_transpose_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv_transpose_1", "inbound_nodes": [[["leaky_relu_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batchnorm_4", "inbound_nodes": [[["conv_transpose_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_relu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_relu_4", "inbound_nodes": [[["batchnorm_4", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv_transpose_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv_transpose_2", "inbound_nodes": [[["leaky_relu_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batchnorm_5", "inbound_nodes": [[["conv_transpose_2", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_relu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_relu_5", "inbound_nodes": [[["batchnorm_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv_transpose_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv_transpose_3", "inbound_nodes": [[["leaky_relu_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batchnorm_6", "inbound_nodes": [[["conv_transpose_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_relu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_relu_6", "inbound_nodes": [[["batchnorm_6", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv_transpose_4", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv_transpose_4", "inbound_nodes": [[["leaky_relu_6", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["conv_transpose_4", 0, 0]]}}, "training_config": {"loss": "SSIMLoss", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0005000000237487257, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_layer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}}
?	

kernel
bias
regularization_losses
	variables
trainable_variables
 	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 3]}}
?	
!axis
	"gamma
#beta
$moving_mean
%moving_variance
&regularization_losses
'	variables
(trainable_variables
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batchnorm_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batchnorm_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 32]}}
?
*regularization_losses
+	variables
,trainable_variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_relu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_relu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?	

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 32]}}
?	
4axis
	5gamma
6beta
7moving_mean
8moving_variance
9regularization_losses
:	variables
;trainable_variables
<	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batchnorm_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batchnorm_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
?
=regularization_losses
>	variables
?trainable_variables
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_relu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_relu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?	

Akernel
Bbias
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
?	
Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batchnorm_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batchnorm_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
?
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_relu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_relu_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?


Tkernel
Ubias
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_transpose_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
?	
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
_regularization_losses
`	variables
atrainable_variables
b	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batchnorm_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batchnorm_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
?
cregularization_losses
d	variables
etrainable_variables
f	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_relu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_relu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?


gkernel
hbias
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv_transpose_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_transpose_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
?	
maxis
	ngamma
obeta
pmoving_mean
qmoving_variance
rregularization_losses
s	variables
ttrainable_variables
u	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batchnorm_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batchnorm_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
?
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_relu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_relu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?


zkernel
{bias
|regularization_losses
}	variables
~trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv_transpose_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_transpose_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batchnorm_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batchnorm_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 32]}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_relu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_relu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?

?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv_transpose_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_transpose_4", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 32]}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem?m?"m?#m?.m?/m?5m?6m?Am?Bm?Hm?Im?Tm?Um?[m?\m?gm?hm?nm?om?zm?{m?	?m?	?m?	?m?	?m?v?v?"v?#v?.v?/v?5v?6v?Av?Bv?Hv?Iv?Tv?Uv?[v?\v?gv?hv?nv?ov?zv?{v?	?v?	?v?	?v?	?v?"
	optimizer
 "
trackable_list_wrapper
?
0
1
"2
#3
$4
%5
.6
/7
58
69
710
811
A12
B13
H14
I15
J16
K17
T18
U19
[20
\21
]22
^23
g24
h25
n26
o27
p28
q29
z30
{31
?32
?33
?34
?35
?36
?37"
trackable_list_wrapper
?
0
1
"2
#3
.4
/5
56
67
A8
B9
H10
I11
T12
U13
[14
\15
g16
h17
n18
o19
z20
{21
?22
?23
?24
?25"
trackable_list_wrapper
?
?non_trainable_variables
regularization_losses
	variables
 ?layer_regularization_losses
?metrics
trainable_variables
?layer_metrics
?layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
':% 2conv_1/kernel
: 2conv_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
?non_trainable_variables
regularization_losses
	variables
 ?layer_regularization_losses
?metrics
trainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: 2batchnorm_1/gamma
: 2batchnorm_1/beta
':%  (2batchnorm_1/moving_mean
+:)  (2batchnorm_1/moving_variance
 "
trackable_list_wrapper
<
"0
#1
$2
%3"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
?non_trainable_variables
&regularization_losses
'	variables
 ?layer_regularization_losses
?metrics
(trainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
*regularization_losses
+	variables
 ?layer_regularization_losses
?metrics
,trainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':% @2conv_2/kernel
:@2conv_2/bias
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
?non_trainable_variables
0regularization_losses
1	variables
 ?layer_regularization_losses
?metrics
2trainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2batchnorm_2/gamma
:@2batchnorm_2/beta
':%@ (2batchnorm_2/moving_mean
+:)@ (2batchnorm_2/moving_variance
 "
trackable_list_wrapper
<
50
61
72
83"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
?
?non_trainable_variables
9regularization_losses
:	variables
 ?layer_regularization_losses
?metrics
;trainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
=regularization_losses
>	variables
 ?layer_regularization_losses
?metrics
?trainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%@@2conv_3/kernel
:@2conv_3/bias
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
?
?non_trainable_variables
Cregularization_losses
D	variables
 ?layer_regularization_losses
?metrics
Etrainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2batchnorm_3/gamma
:@2batchnorm_3/beta
':%@ (2batchnorm_3/moving_mean
+:)@ (2batchnorm_3/moving_variance
 "
trackable_list_wrapper
<
H0
I1
J2
K3"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
?
?non_trainable_variables
Lregularization_losses
M	variables
 ?layer_regularization_losses
?metrics
Ntrainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Pregularization_losses
Q	variables
 ?layer_regularization_losses
?metrics
Rtrainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/@@2conv_transpose_1/kernel
#:!@2conv_transpose_1/bias
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
?
?non_trainable_variables
Vregularization_losses
W	variables
 ?layer_regularization_losses
?metrics
Xtrainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2batchnorm_4/gamma
:@2batchnorm_4/beta
':%@ (2batchnorm_4/moving_mean
+:)@ (2batchnorm_4/moving_variance
 "
trackable_list_wrapper
<
[0
\1
]2
^3"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
?
?non_trainable_variables
_regularization_losses
`	variables
 ?layer_regularization_losses
?metrics
atrainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
cregularization_losses
d	variables
 ?layer_regularization_losses
?metrics
etrainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/@@2conv_transpose_2/kernel
#:!@2conv_transpose_2/bias
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
?
?non_trainable_variables
iregularization_losses
j	variables
 ?layer_regularization_losses
?metrics
ktrainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2batchnorm_5/gamma
:@2batchnorm_5/beta
':%@ (2batchnorm_5/moving_mean
+:)@ (2batchnorm_5/moving_variance
 "
trackable_list_wrapper
<
n0
o1
p2
q3"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
?
?non_trainable_variables
rregularization_losses
s	variables
 ?layer_regularization_losses
?metrics
ttrainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
vregularization_losses
w	variables
 ?layer_regularization_losses
?metrics
xtrainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/ @2conv_transpose_3/kernel
#:! 2conv_transpose_3/bias
 "
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
?
?non_trainable_variables
|regularization_losses
}	variables
 ?layer_regularization_losses
?metrics
~trainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: 2batchnorm_6/gamma
: 2batchnorm_6/beta
':%  (2batchnorm_6/moving_mean
+:)  (2batchnorm_6/moving_variance
 "
trackable_list_wrapper
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
?
?non_trainable_variables
?regularization_losses
?	variables
 ?layer_regularization_losses
?metrics
?trainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?regularization_losses
?	variables
 ?layer_regularization_losses
?metrics
?trainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/ 2conv_transpose_4/kernel
#:!2conv_transpose_4/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?non_trainable_variables
?regularization_losses
?	variables
 ?layer_regularization_losses
?metrics
?trainable_variables
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
x
$0
%1
72
83
J4
K5
]6
^7
p8
q9
?10
?11"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
19"
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
.
$0
%1"
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
 "
trackable_list_wrapper
.
70
81"
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
 "
trackable_list_wrapper
.
J0
K1"
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
 "
trackable_list_wrapper
.
]0
^1"
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
 "
trackable_list_wrapper
.
p0
q1"
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
 "
trackable_list_wrapper
0
?0
?1"
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
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
,:* 2Adam/conv_1/kernel/m
: 2Adam/conv_1/bias/m
$:" 2Adam/batchnorm_1/gamma/m
#:! 2Adam/batchnorm_1/beta/m
,:* @2Adam/conv_2/kernel/m
:@2Adam/conv_2/bias/m
$:"@2Adam/batchnorm_2/gamma/m
#:!@2Adam/batchnorm_2/beta/m
,:*@@2Adam/conv_3/kernel/m
:@2Adam/conv_3/bias/m
$:"@2Adam/batchnorm_3/gamma/m
#:!@2Adam/batchnorm_3/beta/m
6:4@@2Adam/conv_transpose_1/kernel/m
(:&@2Adam/conv_transpose_1/bias/m
$:"@2Adam/batchnorm_4/gamma/m
#:!@2Adam/batchnorm_4/beta/m
6:4@@2Adam/conv_transpose_2/kernel/m
(:&@2Adam/conv_transpose_2/bias/m
$:"@2Adam/batchnorm_5/gamma/m
#:!@2Adam/batchnorm_5/beta/m
6:4 @2Adam/conv_transpose_3/kernel/m
(:& 2Adam/conv_transpose_3/bias/m
$:" 2Adam/batchnorm_6/gamma/m
#:! 2Adam/batchnorm_6/beta/m
6:4 2Adam/conv_transpose_4/kernel/m
(:&2Adam/conv_transpose_4/bias/m
,:* 2Adam/conv_1/kernel/v
: 2Adam/conv_1/bias/v
$:" 2Adam/batchnorm_1/gamma/v
#:! 2Adam/batchnorm_1/beta/v
,:* @2Adam/conv_2/kernel/v
:@2Adam/conv_2/bias/v
$:"@2Adam/batchnorm_2/gamma/v
#:!@2Adam/batchnorm_2/beta/v
,:*@@2Adam/conv_3/kernel/v
:@2Adam/conv_3/bias/v
$:"@2Adam/batchnorm_3/gamma/v
#:!@2Adam/batchnorm_3/beta/v
6:4@@2Adam/conv_transpose_1/kernel/v
(:&@2Adam/conv_transpose_1/bias/v
$:"@2Adam/batchnorm_4/gamma/v
#:!@2Adam/batchnorm_4/beta/v
6:4@@2Adam/conv_transpose_2/kernel/v
(:&@2Adam/conv_transpose_2/bias/v
$:"@2Adam/batchnorm_5/gamma/v
#:!@2Adam/batchnorm_5/beta/v
6:4 @2Adam/conv_transpose_3/kernel/v
(:& 2Adam/conv_transpose_3/bias/v
$:" 2Adam/batchnorm_6/gamma/v
#:! 2Adam/batchnorm_6/beta/v
6:4 2Adam/conv_transpose_4/kernel/v
(:&2Adam/conv_transpose_4/bias/v
?2?
__inference__wrapped_model_4408?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *4?1
/?,
input_layer???????????
?2?
?__inference_model_layer_call_and_return_conditional_losses_5714
?__inference_model_layer_call_and_return_conditional_losses_6655
?__inference_model_layer_call_and_return_conditional_losses_5813
?__inference_model_layer_call_and_return_conditional_losses_6466?
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
?2?
$__inference_model_layer_call_fn_6174
$__inference_model_layer_call_fn_6736
$__inference_model_layer_call_fn_6817
$__inference_model_layer_call_fn_5994?
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
?2?
@__inference_conv_1_layer_call_and_return_conditional_losses_6827?
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
%__inference_conv_1_layer_call_fn_6836?
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
?2?
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_6920
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_6938
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_6856
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_6874?
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
*__inference_batchnorm_1_layer_call_fn_6887
*__inference_batchnorm_1_layer_call_fn_6951
*__inference_batchnorm_1_layer_call_fn_6964
*__inference_batchnorm_1_layer_call_fn_6900?
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
?2?
F__inference_leaky_relu_1_layer_call_and_return_conditional_losses_6969?
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
+__inference_leaky_relu_1_layer_call_fn_6974?
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
@__inference_conv_2_layer_call_and_return_conditional_losses_6984?
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
%__inference_conv_2_layer_call_fn_6993?
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
?2?
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_7031
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_7013
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_7077
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_7095?
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
*__inference_batchnorm_2_layer_call_fn_7057
*__inference_batchnorm_2_layer_call_fn_7108
*__inference_batchnorm_2_layer_call_fn_7044
*__inference_batchnorm_2_layer_call_fn_7121?
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
?2?
F__inference_leaky_relu_2_layer_call_and_return_conditional_losses_7126?
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
+__inference_leaky_relu_2_layer_call_fn_7131?
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
@__inference_conv_3_layer_call_and_return_conditional_losses_7141?
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
%__inference_conv_3_layer_call_fn_7150?
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
?2?
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_7188
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_7170
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_7234
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_7252?
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
*__inference_batchnorm_3_layer_call_fn_7201
*__inference_batchnorm_3_layer_call_fn_7278
*__inference_batchnorm_3_layer_call_fn_7214
*__inference_batchnorm_3_layer_call_fn_7265?
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
?2?
F__inference_leaky_relu_3_layer_call_and_return_conditional_losses_7283?
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
+__inference_leaky_relu_3_layer_call_fn_7288?
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
?2?
J__inference_conv_transpose_1_layer_call_and_return_conditional_losses_4754?
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
annotations? *7?4
2?/+???????????????????????????@
?2?
/__inference_conv_transpose_1_layer_call_fn_4764?
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
annotations? *7?4
2?/+???????????????????????????@
?2?
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_7308
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_7326?
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
*__inference_batchnorm_4_layer_call_fn_7339
*__inference_batchnorm_4_layer_call_fn_7352?
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
?2?
F__inference_leaky_relu_4_layer_call_and_return_conditional_losses_7357?
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
+__inference_leaky_relu_4_layer_call_fn_7362?
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
?2?
J__inference_conv_transpose_2_layer_call_and_return_conditional_losses_4902?
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
annotations? *7?4
2?/+???????????????????????????@
?2?
/__inference_conv_transpose_2_layer_call_fn_4912?
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
annotations? *7?4
2?/+???????????????????????????@
?2?
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_7382
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_7400?
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
*__inference_batchnorm_5_layer_call_fn_7413
*__inference_batchnorm_5_layer_call_fn_7426?
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
?2?
F__inference_leaky_relu_5_layer_call_and_return_conditional_losses_7431?
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
+__inference_leaky_relu_5_layer_call_fn_7436?
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
?2?
J__inference_conv_transpose_3_layer_call_and_return_conditional_losses_5050?
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
annotations? *7?4
2?/+???????????????????????????@
?2?
/__inference_conv_transpose_3_layer_call_fn_5060?
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
annotations? *7?4
2?/+???????????????????????????@
?2?
E__inference_batchnorm_6_layer_call_and_return_conditional_losses_7474
E__inference_batchnorm_6_layer_call_and_return_conditional_losses_7456?
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
*__inference_batchnorm_6_layer_call_fn_7487
*__inference_batchnorm_6_layer_call_fn_7500?
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
?2?
F__inference_leaky_relu_6_layer_call_and_return_conditional_losses_7505?
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
+__inference_leaky_relu_6_layer_call_fn_7510?
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
?2?
J__inference_conv_transpose_4_layer_call_and_return_conditional_losses_5199?
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
annotations? *7?4
2?/+??????????????????????????? 
?2?
/__inference_conv_transpose_4_layer_call_fn_5209?
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
annotations? *7?4
2?/+??????????????????????????? 
?B?
"__inference_signature_wrapper_6265input_layer"?
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
 ?
__inference__wrapped_model_4408?,"#$%./5678ABHIJKTU[\]^ghnopqz{??????>?;
4?1
/?,
input_layer???????????
? "M?J
H
conv_transpose_44?1
conv_transpose_4????????????
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_6856v"#$%=?:
3?0
*?'
inputs??????????? 
p
? "/?,
%?"
0??????????? 
? ?
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_6874v"#$%=?:
3?0
*?'
inputs??????????? 
p 
? "/?,
%?"
0??????????? 
? ?
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_6920?"#$%M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
E__inference_batchnorm_1_layer_call_and_return_conditional_losses_6938?"#$%M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
*__inference_batchnorm_1_layer_call_fn_6887i"#$%=?:
3?0
*?'
inputs??????????? 
p
? ""???????????? ?
*__inference_batchnorm_1_layer_call_fn_6900i"#$%=?:
3?0
*?'
inputs??????????? 
p 
? ""???????????? ?
*__inference_batchnorm_1_layer_call_fn_6951?"#$%M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
*__inference_batchnorm_1_layer_call_fn_6964?"#$%M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_7013?5678M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_7031?5678M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_7077r5678;?8
1?.
(?%
inputs?????????@@@
p
? "-?*
#? 
0?????????@@@
? ?
E__inference_batchnorm_2_layer_call_and_return_conditional_losses_7095r5678;?8
1?.
(?%
inputs?????????@@@
p 
? "-?*
#? 
0?????????@@@
? ?
*__inference_batchnorm_2_layer_call_fn_7044?5678M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
*__inference_batchnorm_2_layer_call_fn_7057?5678M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
*__inference_batchnorm_2_layer_call_fn_7108e5678;?8
1?.
(?%
inputs?????????@@@
p
? " ??????????@@@?
*__inference_batchnorm_2_layer_call_fn_7121e5678;?8
1?.
(?%
inputs?????????@@@
p 
? " ??????????@@@?
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_7170?HIJKM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_7188?HIJKM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_7234rHIJK;?8
1?.
(?%
inputs?????????  @
p
? "-?*
#? 
0?????????  @
? ?
E__inference_batchnorm_3_layer_call_and_return_conditional_losses_7252rHIJK;?8
1?.
(?%
inputs?????????  @
p 
? "-?*
#? 
0?????????  @
? ?
*__inference_batchnorm_3_layer_call_fn_7201?HIJKM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
*__inference_batchnorm_3_layer_call_fn_7214?HIJKM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
*__inference_batchnorm_3_layer_call_fn_7265eHIJK;?8
1?.
(?%
inputs?????????  @
p
? " ??????????  @?
*__inference_batchnorm_3_layer_call_fn_7278eHIJK;?8
1?.
(?%
inputs?????????  @
p 
? " ??????????  @?
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_7308?[\]^M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
E__inference_batchnorm_4_layer_call_and_return_conditional_losses_7326?[\]^M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
*__inference_batchnorm_4_layer_call_fn_7339?[\]^M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
*__inference_batchnorm_4_layer_call_fn_7352?[\]^M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_7382?nopqM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
E__inference_batchnorm_5_layer_call_and_return_conditional_losses_7400?nopqM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
*__inference_batchnorm_5_layer_call_fn_7413?nopqM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
*__inference_batchnorm_5_layer_call_fn_7426?nopqM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
E__inference_batchnorm_6_layer_call_and_return_conditional_losses_7456?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
E__inference_batchnorm_6_layer_call_and_return_conditional_losses_7474?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
*__inference_batchnorm_6_layer_call_fn_7487?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
*__inference_batchnorm_6_layer_call_fn_7500?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
@__inference_conv_1_layer_call_and_return_conditional_losses_6827p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0??????????? 
? ?
%__inference_conv_1_layer_call_fn_6836c9?6
/?,
*?'
inputs???????????
? ""???????????? ?
@__inference_conv_2_layer_call_and_return_conditional_losses_6984n./9?6
/?,
*?'
inputs??????????? 
? "-?*
#? 
0?????????@@@
? ?
%__inference_conv_2_layer_call_fn_6993a./9?6
/?,
*?'
inputs??????????? 
? " ??????????@@@?
@__inference_conv_3_layer_call_and_return_conditional_losses_7141lAB7?4
-?*
(?%
inputs?????????@@@
? "-?*
#? 
0?????????  @
? ?
%__inference_conv_3_layer_call_fn_7150_AB7?4
-?*
(?%
inputs?????????@@@
? " ??????????  @?
J__inference_conv_transpose_1_layer_call_and_return_conditional_losses_4754?TUI?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
/__inference_conv_transpose_1_layer_call_fn_4764?TUI?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
J__inference_conv_transpose_2_layer_call_and_return_conditional_losses_4902?ghI?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
/__inference_conv_transpose_2_layer_call_fn_4912?ghI?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
J__inference_conv_transpose_3_layer_call_and_return_conditional_losses_5050?z{I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
/__inference_conv_transpose_3_layer_call_fn_5060?z{I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
J__inference_conv_transpose_4_layer_call_and_return_conditional_losses_5199???I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
/__inference_conv_transpose_4_layer_call_fn_5209???I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
F__inference_leaky_relu_1_layer_call_and_return_conditional_losses_6969l9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0??????????? 
? ?
+__inference_leaky_relu_1_layer_call_fn_6974_9?6
/?,
*?'
inputs??????????? 
? ""???????????? ?
F__inference_leaky_relu_2_layer_call_and_return_conditional_losses_7126h7?4
-?*
(?%
inputs?????????@@@
? "-?*
#? 
0?????????@@@
? ?
+__inference_leaky_relu_2_layer_call_fn_7131[7?4
-?*
(?%
inputs?????????@@@
? " ??????????@@@?
F__inference_leaky_relu_3_layer_call_and_return_conditional_losses_7283h7?4
-?*
(?%
inputs?????????  @
? "-?*
#? 
0?????????  @
? ?
+__inference_leaky_relu_3_layer_call_fn_7288[7?4
-?*
(?%
inputs?????????  @
? " ??????????  @?
F__inference_leaky_relu_4_layer_call_and_return_conditional_losses_7357?I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
+__inference_leaky_relu_4_layer_call_fn_7362I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
F__inference_leaky_relu_5_layer_call_and_return_conditional_losses_7431?I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
+__inference_leaky_relu_5_layer_call_fn_7436I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
F__inference_leaky_relu_6_layer_call_and_return_conditional_losses_7505?I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
+__inference_leaky_relu_6_layer_call_fn_7510I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
?__inference_model_layer_call_and_return_conditional_losses_5714?,"#$%./5678ABHIJKTU[\]^ghnopqz{??????F?C
<?9
/?,
input_layer???????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_5813?,"#$%./5678ABHIJKTU[\]^ghnopqz{??????F?C
<?9
/?,
input_layer???????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_6466?,"#$%./5678ABHIJKTU[\]^ghnopqz{??????A?>
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
?__inference_model_layer_call_and_return_conditional_losses_6655?,"#$%./5678ABHIJKTU[\]^ghnopqz{??????A?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
$__inference_model_layer_call_fn_5994?,"#$%./5678ABHIJKTU[\]^ghnopqz{??????F?C
<?9
/?,
input_layer???????????
p

 
? "2?/+????????????????????????????
$__inference_model_layer_call_fn_6174?,"#$%./5678ABHIJKTU[\]^ghnopqz{??????F?C
<?9
/?,
input_layer???????????
p 

 
? "2?/+????????????????????????????
$__inference_model_layer_call_fn_6736?,"#$%./5678ABHIJKTU[\]^ghnopqz{??????A?>
7?4
*?'
inputs???????????
p

 
? "2?/+????????????????????????????
$__inference_model_layer_call_fn_6817?,"#$%./5678ABHIJKTU[\]^ghnopqz{??????A?>
7?4
*?'
inputs???????????
p 

 
? "2?/+????????????????????????????
"__inference_signature_wrapper_6265?,"#$%./5678ABHIJKTU[\]^ghnopqz{??????M?J
? 
C?@
>
input_layer/?,
input_layer???????????"M?J
H
conv_transpose_44?1
conv_transpose_4???????????