Shader "Unlit/RayTracing/ProceduralSphere"
{
    Properties
    {
        _Color("Main Color", Color) = (1, 1, 1, 1)
        _MainTex("Albedo (RGB)", 2D) = "white" {}
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
            #include "Light.hlsl"
            #include "RayPayload.hlsl"

            #pragma raytracing test

            #pragma multi_compile_local __ RAY_TRACING_PROCEDURAL_GEOMETRY

            float4 _Color;

#if RAY_TRACING_PROCEDURAL_GEOMETRY

            struct AttributeData
            {
                float3 normal;
            };

            [shader("intersection")]
            void ProceduralSphereIntersectionMain()
            {
                const float radius = 0.495;

                float3 o = ObjectRayOrigin();
                float3 d = ObjectRayDirection();
                float a = dot(d, d);
                float b = 2 * dot(o, d);
                float c = dot(o, o) - radius * radius;
                float delta2 = b * b - 4 * a * c;
                if (delta2 >= 0)
                {
                    float t0 = (-b + sqrt(delta2)) / (2 * a);
                    float t1 = (-b - sqrt(delta2)) / (2 * a);

                    // Get the smallest root larger than 0 (t is in object space);
                    float t = max(t0, t1);
                    if (t0 >= 0)
                        t = min(t, t0);
                    if (t1 >= 0)
                        t = min(t, t1);

                    float3 localPos = ObjectRayOrigin() + t * ObjectRayDirection();

                    AttributeData attr;
                    attr.normal = normalize(localPos);

                    float3 worldPos = mul(ObjectToWorld(), float4(localPos, 1));

                    float THit = length(worldPos - WorldRayOrigin());

                    ReportHit(THit, 0, attr);
                }
            }

            [shader("closesthit")]
            void ClosestHitMain(inout RayPayload payload : SV_RayPayload, AttributeData attribs : SV_IntersectionAttributes)
            {      
                float3 worldNormal = normalize(mul(attribs.normal, (float3x3)WorldToObject()));

                float3 worldPosition = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
                
                float3 vecToLight = PointLightPosition.xyz - worldPosition;

                float distToLight = length(PointLightPosition.xyz - worldPosition);

                float3 albedo = _Color.xyz * PointLightColor * PointLightIntensity * saturate(dot(worldNormal, normalize(vecToLight))) * CalculateLightFalloff(distToLight, PointLightRange);

                payload.color = float4(albedo, 1);
                payload.worldPos = float4(worldPosition, 1);
            }
#else
            struct AttributeData
            {
                float2 bary;
            };

            [shader("closesthit")]
            void ClosestHitMain(inout RayPayload payload : SV_RayPayload, AttributeData attribs : SV_IntersectionAttributes)
            {
                payload.color = _Color;
                payload.worldPos = float4(WorldRayOrigin() + RayTCurrent() * WorldRayDirection(), 1);
            }
#endif

            ENDHLSL
        }
    }
}
