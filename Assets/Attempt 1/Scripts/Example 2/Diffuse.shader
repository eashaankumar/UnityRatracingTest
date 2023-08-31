Shader "Unlit/RayTracing/Diffuse"
{
    Properties
    {
        _Color("Main Color", Color) = (1, 1, 1, 1)
        _MainTex("Albedo (RGB)", 2D) = "white" {}
        _Energy("Energy", Float) = 1
    }

    SubShader
    {
        Tags { "RenderType" = "Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            float4 _Color;

            struct appdata
            {
                float4 vertex : POSITION;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                fixed4 col = _Color;

                return col;
            }
            ENDCG
        }
    }

    SubShader
    {
        Pass
        {
            Name "Test"
            Tags{ "LightMode" = "RayTracing" }

            HLSLPROGRAM

            #include "UnityShaderVariables.cginc"
            #include "UnityRaytracingMeshUtils.cginc"
            #include "Light.hlsl"
            #include "RayPayload.hlsl"

            #pragma raytracing test

            float4 _Color;    
            float _Energy;

            Texture2D _MainTex;
            float4 _MainTex_ST;

            SamplerState sampler_linear_repeat;

            RaytracingAccelerationStructure g_SceneAccelStruct;

            struct AttributeData
            {
                float2 barycentrics;
            };

            struct Vertex
            {
                float3 position;
                float3 normal;
                float2 uv;
            };

            Vertex FetchVertex(uint vertexIndex)
            {
                Vertex v;
                v.position  = UnityRayTracingFetchVertexAttribute3(vertexIndex, kVertexAttributePosition);
                v.normal    = UnityRayTracingFetchVertexAttribute3(vertexIndex, kVertexAttributeNormal);
                v.uv        = UnityRayTracingFetchVertexAttribute2(vertexIndex, kVertexAttributeTexCoord0);
                return v;
            }

            Vertex InterpolateVertices(Vertex v0, Vertex v1, Vertex v2, float3 barycentrics)
            {
                Vertex v;
                #define INTERPOLATE_ATTRIBUTE(attr) v.attr = v0.attr * barycentrics.x + v1.attr * barycentrics.y + v2.attr * barycentrics.z
                INTERPOLATE_ATTRIBUTE(position);
                INTERPOLATE_ATTRIBUTE(normal);
                INTERPOLATE_ATTRIBUTE(uv);
                return v;
            }

            [shader("closesthit")]
            void ClosestHitMain(inout RayPayload payload : SV_RayPayload, AttributeData attribs : SV_IntersectionAttributes)
            {
                uint3 triangleIndices = UnityRayTracingFetchTriangleIndices(PrimitiveIndex());

                Vertex v0, v1, v2;
                v0 = FetchVertex(triangleIndices.x);
                v1 = FetchVertex(triangleIndices.y);
                v2 = FetchVertex(triangleIndices.z);

                float3 barycentricCoords = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
                Vertex v = InterpolateVertices(v0, v1, v2, barycentricCoords);

                float3 worldPosition = mul(ObjectToWorld(), float4(v.position, 1));


                float3 e0 = v1.position - v0.position;
                float3 e1 = v2.position - v0.position;

                //      float3 faceNormal = normalize(mul(cross(e0, e1), (float3x3)WorldToObject()));
                float3 faceNormal = normalize(mul(v.normal, (float3x3)WorldToObject()));

                bool isFrontFace = (HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE);
                faceNormal = (isFrontFace == false) ? -faceNormal : faceNormal;

                // float3 reflectedRay = reflect(WorldRayDirection(), faceNormal); // perfect reflection
                float3 reflectedRay = RandomHemisphereDirection(faceNormal, worldPosition.x);

                payload.worldPos = worldPosition + 0.01f * reflectedRay;
                payload.worldDir = reflectedRay;

                float4 emittedLight = _Color * _Energy;
                payload.energy += emittedLight * payload.color;
                payload.color *= _Color;

                /*RayDesc ray;

                ray.Origin = worldPosition + 0.01f * reflectedRay;
                ray.Direction = reflectedRay;
                ray.TMin = 0;
                ray.TMax = 1e20f;*/

                //payload.bounceIndex += 1;

                //TraceRay(g_SceneAccelStruct, 0, 0xFF, 0, 1, 0, ray, payload);


                /*RayPayload reflRayPayload;
                reflRayPayload.color = payload.color;
                reflRayPayload.worldPos = float4(0, 0, 0, 1);
                reflRayPayload.bounceIndex = payload.bounceIndex + 1;
                reflRayPayload.energy = payload.energy;

                TraceRay(g_SceneAccelStruct, 0, 0xFF, 0, 1, 0, ray, reflRayPayload);

                float3 vecToLight = PointLightPosition.xyz - worldPosition;
                float distToLight = length(PointLightPosition.xyz - worldPosition);

                float3 texColor = _MainTex.SampleLevel(sampler_linear_repeat, v.uv * _MainTex_ST.xy, 0).rgb;

                float3 albedo = texColor * _Color.xyz * PointLightColor * PointLightIntensity * saturate(dot(faceNormal, normalize(vecToLight))) * CalculateLightFalloff(distToLight, PointLightRange);

                payload.color = float4(albedo, 1);
                payload.worldPos = float4(worldPosition, 1);
                payload.energy = _Energy;*/

                /*if (payload.bounceIndex == 0)
                {
                    payload.worldPos = float4(worldPosition, 1);
                }*/
            
            }

            ENDHLSL
        }
    }
}
