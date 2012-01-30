#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=g++
CXX=g++
FC=gfortran
AS=as

# Macros
CND_PLATFORM=GNU-Linux-x86
CND_CONF=Debug
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/keller.o \
	${OBJECTDIR}/conus_projection_sparse.o \
	${OBJECTDIR}/sparce_types.o \
	${OBJECTDIR}/sparse_operations_ext.o \
	${OBJECTDIR}/simplex_projection.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=-m64 -O2 -g
CXXFLAGS=-m64 -O2 -g

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=-latlas -lblas -llapack -lm -lldl

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/projection

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/projection: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/projection ${OBJECTFILES} ${LDLIBSOPTIONS} 

${OBJECTDIR}/keller.o: keller.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -O2 -I/usr/include -MMD -MP -MF $@.d -o ${OBJECTDIR}/keller.o keller.cpp

${OBJECTDIR}/conus_projection_sparse.o: conus_projection_sparse.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -O2 -I/usr/include -MMD -MP -MF $@.d -o ${OBJECTDIR}/conus_projection_sparse.o conus_projection_sparse.cpp

${OBJECTDIR}/sparce_types.o: sparce_types.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -O2 -I/usr/include -MMD -MP -MF $@.d -o ${OBJECTDIR}/sparce_types.o sparce_types.cpp

${OBJECTDIR}/sparse_operations_ext.o: sparse_operations_ext.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -O2 -I/usr/include -MMD -MP -MF $@.d -o ${OBJECTDIR}/sparse_operations_ext.o sparse_operations_ext.cpp

${OBJECTDIR}/simplex_projection.o: simplex_projection.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -O2 -I/usr/include -MMD -MP -MF $@.d -o ${OBJECTDIR}/simplex_projection.o simplex_projection.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/projection

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
