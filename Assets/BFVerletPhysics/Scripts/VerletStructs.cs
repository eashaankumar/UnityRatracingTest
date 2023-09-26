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

        public VoxelWorld(Allocator a)
        {
            voxels = new NativeList<Voxel>(a);
        }

        public void Dispose()
        {
            if (voxels.IsCreated) voxels.Dispose();
        }

        public Voxel GetVoxel(int i)
        {
            return voxels[i];
        }

        public void AddVoxel(Voxel v)
        {
            if (voxels.IsCreated) voxels.Add(v);
        }
    }

    public struct Data
    {
        public float3 color;
        public float3 specular;
        public float3 smoothness;
        public float3 metallic;
        public float3 emission;
        public float ior;

        public static int Size
        {
            get
            {
                return (3 + 3 + 3 + 3 + 3 + 1) * sizeof(float);
            }
        }
    };

    public struct VerletPhysicsRenderer : System.IDisposable
    {
        public NativeArray<Data> data;
        public NativeArray<Matrix4x4> matrices;

        public VerletPhysicsRenderer(int count, Allocator alloc)
        {
            data = new NativeArray<Data>(count, alloc);
            matrices = new NativeArray<Matrix4x4>(count, alloc);
        }

        public void Dispose()
        {
            if (data.IsCreated) data.Dispose();
            if (matrices.IsCreated) matrices.Dispose();
        }
    }

    public struct Color
    {
        public byte r, g, b;

        public Color (float _r, float _g, float _b)
        {
            r = (byte)(_r * 255);
            g = (byte)(_g * 255);
            b = (byte)(_b * 255);
        }

        public float3 Float3()
        {
            return new float3(r / 255f, g / 255f, b / 255f);
        }
    }

    public struct Voxel
    {
        public Color color;
        public byte glow;

        public Voxel(Color _c, float _g)
        {
            color = _c;
            glow = (byte)(_g * 255);
        }
    }
}

namespace BarelyFunctional.Interfaces
{
    public interface IVerletPhysicsRenderer
    {
        public void SetRenderer(VerletPhysicsRenderer readyToBeCopied);
    }
}
