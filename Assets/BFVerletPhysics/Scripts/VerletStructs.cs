using BarelyFunctional.Interfaces;
using BarelyFunctional.Structs;
using System.Collections;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace BarelyFunctional.Structs
{
    // https://en.wikipedia.org/wiki/Verlet_integration
    [System.Serializable]
    public struct VerletParticle
    {
        public double3 pos_current;
        public double3 pos_old;
        public double3 acc;
        public double radius;

        public double mass; // 1kg
        public double drag; // rho*C*Area – simplified drag for this example
        public bool freeze;

        public VerletParticle(double _mass, double _drag, double _r, bool _freeze)
        {
            this.pos_current = 0;
            this.pos_old = 0;
            this.acc = 0;
            this.mass = _mass;
            this.drag = _drag;
            this.radius = _r;
            this.freeze = _freeze;
        }

        public void update(double dt)
        {
            /*double3 new_pos = pos + vel * dt + acc * (dt * dt * 0.5);
            double3 new_acc = gravityAcc + dragAcc();
            double3 new_vel = vel + (acc + new_acc) * (dt * 0.5);
            pos = new_pos;
            vel = new_vel;
            acc = new_acc;*/
            double3 vel = pos_current - pos_old;
            pos_old = pos_current;
            pos_current = pos_current + vel + acc * dt * dt;
            acc = 0;
        }

        public void accelerate(double3 externalAcc)
        {
            acc += externalAcc;
        }

        /*double3 dragAcc()
        {
            // double3 grav_acc = new double3 ( 0.0, 0.0, -9.81 ); // 9.81 m/s² down in the z-axis
            double3 drag_force = 0.5 * drag * (vel * vel); // D = 0.5 * (rho * C * Area * vel^2)
            double3 drag_acc = drag_force / mass; // a = F/m
            return /*grav_acc*/ /*-drag_acc;
        }*/

        public int3 Hash(double cellSize)
        {
            return new int3(
                (int)math.floor(pos_current.x / cellSize),
                (int)math.floor(pos_current.y / cellSize),
                (int)math.floor(pos_current.z / cellSize)
                );
        }
    }

    public struct VerletTargetDistLink : IEqualityComparer<VerletTargetDistLink>
    {
        public int p1Id, p2Id;
        public double targetDist;
        public VerletTargetDistLink(int _p1, int _p2, double _targetDist)
        {
            p1Id = _p1;
            p2Id = _p2;
            targetDist = _targetDist;
        }

        public bool Equals(VerletTargetDistLink x, VerletTargetDistLink y)
        {
            return x.p1Id == y.p1Id && x.p2Id == y.p2Id;
        }

        public int GetHashCode(VerletTargetDistLink obj)
        {
            return obj.p1Id.GetHashCode() + obj.p2Id.GetHashCode();
        }

        public void Apply(VerletPhysicsWorld world)
        {
            VerletParticle p1 = world.GetParticle(p1Id);
            VerletParticle p2 = world.GetParticle(p2Id);

            double3 axis = p1.pos_current - p2.pos_current;
            double dist = math.length(axis);
            double3 n = axis / dist;
            double delta = targetDist - dist;
            p1.pos_current += 0.5 * delta * n;
            p2.pos_current -= 0.5 * delta * n;

            world.SetParticle(p1Id, p1);
            world.SetParticle(p2Id, p2);
        }
    }

    public struct VerletPhysicsWorld : System.IDisposable
    {
        public NativeList<VerletParticle> particles;
        public double3 worldGravity;
        bool isCreated;

        public bool IsCreated
        {
            get { return isCreated; }
        }

        public VerletPhysicsWorld(double3 g, Allocator a)
        {
            particles = new NativeList<VerletParticle>(a);
            worldGravity = g;
            isCreated = true;
        }

        public void AddParticle(VerletParticle p)
        {
            if (!particles.IsCreated) return;
            particles.Add(p);
        }

        public int ParticleCount
        {
            get
            {
                return particles.IsCreated ? particles.Length : 0;
            }
        }

        public VerletParticle GetParticle(int i)
        {
            return particles[i];
        }

        public void SetParticle(int i, VerletParticle p)
        {
            particles[i] = p;
        }

        public void Dispose()
        {
            if (particles.IsCreated) particles.Dispose();
            isCreated = false;
        }
    }

    public struct VoxelWorld : System.IDisposable
    {
        NativeList<Voxel> voxels;
        NativeHashMap<uint, StandardMaterialData> standardMaterialData;
        NativeHashMap<uint, GlassMaterialData> glassMaterialData;
        Unity.Mathematics.Random idGenerator;

        public VoxelWorld(uint seed, Allocator a)
        {
            voxels = new NativeList<Voxel>(a);
            glassMaterialData = new NativeHashMap<uint, GlassMaterialData>(1000000, a);
            standardMaterialData = new NativeHashMap<uint, StandardMaterialData>(1000000, a);
            idGenerator = new Unity.Mathematics.Random(seed);
        }

        public void Dispose()
        {
            if (voxels.IsCreated) voxels.Dispose();
            if (glassMaterialData.IsCreated) glassMaterialData.Dispose();
            if (standardMaterialData.IsCreated) standardMaterialData.Dispose();
        }

        public Voxel GetVoxel(int i)
        {
            return voxels[i];
        }

        public int GlassVoxels
        {
            get
            {
                return glassMaterialData.Count();
            }
        }

        public int StandardVoxels
        {
            get
            {
                return standardMaterialData.Count();
            }
        }

        public StandardMaterialData GetStandardMaterial(uint voxel)
        {
            return standardMaterialData[voxel];
        }

        public GlassMaterialData GetGlassMaterial(uint voxel)
        {
            return glassMaterialData[voxel];
        }

        public void AddVoxel(float size, StandardMaterialData material, uint time)
        {
            if (voxels.IsCreated && standardMaterialData.IsCreated)
            {
                uint id = time + idGenerator.NextUInt();
                if (standardMaterialData.TryAdd(id, material))
                {
                    Voxel v = new Voxel
                    {
                        type = VoxelMaterialType.STANDARD,
                        id = id,
                        size = size,
                    };
                    voxels.Add(v);
                }
                else
                {
                    throw new System.Exception($"Standard Material data already exists for voxel {id}");
                }
            }
        }

        public void AddVoxel(float size, GlassMaterialData material, uint time)
        {
            if (voxels.IsCreated && glassMaterialData.IsCreated)
            {
                uint id = time + idGenerator.NextUInt();
                if (glassMaterialData.TryAdd(id, material))
                {
                    Voxel v = new Voxel
                    {
                        type = VoxelMaterialType.GLASS,
                        id = id,
                        size = size,
                    };
                    voxels.Add(v);
                }
                else
                {
                    throw new System.Exception($"Glass Material data already exists for voxel {id}");
                }
            }
        }
    }

    public struct StandardMaterialData
    {
        public float3 albedo;
        public float3 specular;
        public float3 emission;
        public float smoothness;
        public float metallic;
        public float ior;

        public static int Size
        {
            get
            {
                return (3 + 3 + 3 + 1 + 1 + 1) * sizeof(float);
            }
        }
    };

    public struct GlassMaterialData
    {
        public float3 albedo;
        public float3 emission;
        public float ior; // 1.0, 2.8
        public float roughness; // 0, 0.5
        public float extinctionCoeff; // 0, 1 (technically, 0, 20 but that explodes the colors)
        public float flatShading; // bool

        public static int Size
        {
            get
            {
                return (3 + 3 + 1 + 1 + 1 + 1) * sizeof(float);
            }
        }
    }

    public struct StandardVoxelAssembledData
    {
        public StandardMaterialData material;
        public Matrix4x4 trs;
    }

    public struct GlassVoxelAssembledData
    {
        public GlassMaterialData material;
        public Matrix4x4 trs;
    }

    public struct VerletPhysicsRendererAssembler : System.IDisposable
    {
        public NativeList<StandardVoxelAssembledData> standardMaterialAssembly;
        public NativeList<GlassVoxelAssembledData> glassMaterialAssembly;

        public VerletPhysicsRendererAssembler(int standardVoxels, int glassVoxels, Allocator alloc)
        {
            standardMaterialAssembly = new NativeList<StandardVoxelAssembledData>(standardVoxels, alloc);
            glassMaterialAssembly = new NativeList<GlassVoxelAssembledData>(glassVoxels, alloc);
        }

        public void Dispose()
        {
            if (standardMaterialAssembly.IsCreated) standardMaterialAssembly.Dispose();
            if (glassMaterialAssembly.IsCreated) glassMaterialAssembly.Dispose();
        }
    }

    public struct VerletPhysicsRenderer : System.IDisposable
    {
        public NativeArray<StandardMaterialData> standardMaterialData;
        public NativeArray<GlassMaterialData> glassMaterialData;
        public NativeArray<Matrix4x4> standardMatrices;
        public NativeArray<Matrix4x4> glassMatrices;

        public VerletPhysicsRenderer(int standardVoxles, int glassVoxels, Allocator alloc)
        {
            standardMaterialData = new NativeArray<StandardMaterialData>(standardVoxles, alloc);
            glassMaterialData = new NativeArray<GlassMaterialData>(glassVoxels, alloc);
            standardMatrices = new NativeArray<Matrix4x4>(standardVoxles, alloc);
            glassMatrices = new NativeArray<Matrix4x4>(glassVoxels, alloc);
        }

        public void Dispose()
        {
            if (standardMaterialData.IsCreated) standardMaterialData.Dispose();
            if (glassMaterialData.IsCreated) glassMaterialData.Dispose();
            if (standardMatrices.IsCreated) standardMatrices.Dispose();
            if (glassMatrices.IsCreated) glassMatrices.Dispose();   
        }
    }

    public enum VoxelMaterialType
    {
        STANDARD, GLASS
    }

    public struct Voxel
    {
        public VoxelMaterialType type;
        public uint id;
        public float size;
    }
}

namespace BarelyFunctional.Interfaces
{
    public interface IVerletPhysicsRenderer
    {
        public void SetRenderer(VerletPhysicsRenderer readyToBeCopied);
    }
}
