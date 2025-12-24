using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections;
using System;
using Random = UnityEngine.Random;
using UnityEngine.InputSystem;

public class DoggyAgent : Agent
{
    [Header("Сервоприводы")]
    public ArticulationBody[] legs;

    [Header("Скорость работы сервоприводов")]
    public float servoSpeed;

    [Header("Тело")]
    public ArticulationBody body;
    private Vector3 defPos;
    private Quaternion defRot;
    public float strenghtMove;

    [Header("Куб (цель)")]
    public GameObject cube;

    [Header("Сенсоры")]
    public Unity.MLAgentsExamples.GroundContact[] groundContacts;

    private float distToTarget = 0f;

    // ---------------- rewards: thresholds & terminals ----------------
    [Header("Награды: терминалы")]
    public float reachDistance = 1.0f;
    public float reachReward = 4.0f;
    public float fallY = 0.10f;
    public float fallReward = -2.0f;

    // ---------------- rewards: shaping ----------------
    [Header("Награды: прогресс")]
    public float perStepPenalty = 0.0008f;
    public float progressScale = 3.0f;
    public float progressClip = 0.20f;

    // ---------------- rewards: anti-stuck ----------------
    [Header("Награды: анти-стагнация")]
    public int noImproveLimit = 220;
    public float noImprovePenalty = -1.0f;
    public float idlePenalty = 0.01f;
    public float idleSpeed = 0.18f;
    public float idleProgress = 0.004f;

    // ---------------- rewards: stability & motion ----------------
    [Header("Награды: стабилизация")]
    public float tiltPenaltyScale = 0.010f;
    public float slipPenaltyScale = 0.010f;
    public float rollPenaltyScale = 0.020f;
    public float rollRatePenaltyScale = 0.00025f;

    // ---------------- rewards: action regularization ----------------
    [Header("Награды: регуляризация действий")]
    public float actionEnergyScale = 0.00012f;
    public float actionJerkScale = 0.00006f;
    public int jerkGraceSteps = 60;

    private float _prevDistance;
    private float _bestDistance;
    private int _noImproveSteps;
    private int _stepId;
    private float _prevAbsHeadingDeg;
    private float[] _prevActions;

    public override void Initialize()
    {
        distToTarget = Vector3.Distance(body.transform.position, cube.transform.position);
        defRot = body.transform.rotation;
        defPos = body.transform.position;

        _prevActions = new float[12];
        for (int i = 0; i < _prevActions.Length; i++) _prevActions[i] = 0f;

        _prevDistance = distToTarget;
        _bestDistance = distToTarget;
        _noImproveSteps = 0;
        _stepId = 0;

        _prevAbsHeadingDeg = ComputeAbsHeadingDeg();
    }

    public void ResetDog()
    {
        Quaternion newRot = Quaternion.Euler(-90, 0, Random.Range(0f, 360f));

        body.TeleportRoot(defPos, newRot);
        body.velocity = Vector3.zero;
        body.angularVelocity = Vector3.zero;

        for (int i = 0; i < 12; i++)
        {
            MoveLeg(legs[i], 0);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        Debug.Log("Heuristic");
    }

    public override void OnEpisodeBegin()
    {
        ResetDog();

        cube.transform.position = new Vector3(
            Random.Range(-7.5f, 7.5f),
            0.21f,
            Random.Range(-7.5f, 7.5f)
        );

        distToTarget = Vector3.Distance(body.transform.position, cube.transform.position);
        _prevDistance = distToTarget;
        _bestDistance = distToTarget;
        _noImproveSteps = 0;
        _stepId = 0;

        if (_prevActions == null || _prevActions.Length != 12) _prevActions = new float[12];
        for (int i = 0; i < _prevActions.Length; i++) _prevActions[i] = 0f;

        _prevAbsHeadingDeg = ComputeAbsHeadingDeg();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(body.transform.position);
        sensor.AddObservation(body.velocity);
        sensor.AddObservation(body.angularVelocity);
        sensor.AddObservation(body.transform.right);

        sensor.AddObservation(cube.transform.position);

        Vector3 relativePosition = cube.transform.position - body.transform.position;
        sensor.AddObservation(relativePosition);

        Vector3 toCube = (cube.transform.position - body.transform.position).normalized;
        float angleToCube = Vector3.SignedAngle(body.transform.right, toCube, Vector3.up);
        sensor.AddObservation(angleToCube);

        float distanceToCube = Vector3.Distance(body.transform.position, cube.transform.position);
        sensor.AddObservation(distanceToCube);

        foreach (var leg in legs)
        {
            sensor.AddObservation(leg.xDrive.target);
            sensor.AddObservation(leg.velocity);
            sensor.AddObservation(leg.angularVelocity);
        }

        foreach (var groundContact in groundContacts)
        {
            sensor.AddObservation(groundContact.touchingGround);
        }
    }

    public override void OnActionReceived(ActionBuffers vectorAction)
    {
        var actions = vectorAction.ContinuousActions;

        for (int i = 0; i < 12; i++)
        {
            float angle = Mathf.Lerp(
                legs[i].xDrive.lowerLimit,
                legs[i].xDrive.upperLimit,
                (actions[i] + 1) * 0.5f
            );
            MoveLeg(legs[i], angle);
        }

        float currentDistanceToTarget = Vector3.Distance(body.transform.position, cube.transform.position);

        _stepId++;

        if (body.transform.position.y < fallY)
        {
            AddReward(fallReward);
            EndEpisode();
            return;
        }

        if (currentDistanceToTarget <= reachDistance)
        {
            AddReward(reachReward);
            EndEpisode();
            return;
        }

        const float improveEps = 0.02f;
        if (currentDistanceToTarget < _bestDistance - improveEps)
        {
            _bestDistance = currentDistanceToTarget;
            _noImproveSteps = 0;
            AddReward(0.08f);
        }
        else
        {
            _noImproveSteps++;
            if (_noImproveSteps >= noImproveLimit)
            {
                AddReward(noImprovePenalty);
                EndEpisode();
                return;
            }
        }

        float rawProgress = _prevDistance - currentDistanceToTarget;
        float clipped = Mathf.Clamp(rawProgress, -progressClip, progressClip);
        AddReward(clipped * progressScale);

        AddReward(-perStepPenalty);

        if (body.velocity.magnitude < idleSpeed && Mathf.Abs(rawProgress) < idleProgress)
        {
            AddReward(-idlePenalty);
        }

        float upAlign = Vector3.Dot(body.transform.up, Vector3.up);
        float tilt01 = Mathf.Clamp01(1f - Mathf.Max(0f, upAlign));
        AddReward(-tilt01 * tiltPenaltyScale);

        Vector3 toTarget = cube.transform.position - body.transform.position;
        if (toTarget.sqrMagnitude > 1e-6f)
        {
            Vector3 dir = toTarget.normalized;

            float along = Vector3.Dot(body.velocity, dir);
            Vector3 lateral = body.velocity - along * dir;
            AddReward(-lateral.sqrMagnitude * slipPenaltyScale);

            float absHeading = AbsSignedAngleDeg(body.transform.right, dir, Vector3.up);
            float deltaHeading = (_prevAbsHeadingDeg - absHeading) / 180f;
            AddReward(deltaHeading * 0.02f);
            _prevAbsHeadingDeg = absHeading;

            Vector3 fwd = body.transform.right;
            Vector3 upProj = Vector3.ProjectOnPlane(body.transform.up, fwd);
            Vector3 worldUpProj = Vector3.ProjectOnPlane(Vector3.up, fwd);

            if (upProj.sqrMagnitude > 1e-6f && worldUpProj.sqrMagnitude > 1e-6f)
            {
                upProj.Normalize();
                worldUpProj.Normalize();
                float rollDeg = Vector3.Angle(worldUpProj, upProj);

                float rollT = Mathf.Clamp01((rollDeg - 10f) / 45f);
                AddReward(-rollT * rollPenaltyScale);
            }

            float rollRate = Mathf.Abs(Vector3.Dot(body.angularVelocity, fwd));
            AddReward(-rollRate * rollRatePenaltyScale);
        }

        int contacts = 0;
        if (groundContacts != null)
        {
            for (int i = 0; i < groundContacts.Length; i++)
                if (groundContacts[i] != null && groundContacts[i].touchingGround) contacts++;
        }

        if (contacts >= 2) AddReward(0.0015f);
        else if (contacts == 0) AddReward(-0.0020f);

        float energy = 0f;
        float jerk = 0f;

        for (int i = 0; i < 12; i++)
        {
            float a = Mathf.Clamp(actions[i], -1f, 1f);
            energy += a * a;

            float da = a - _prevActions[i];
            jerk += da * da;

            _prevActions[i] = a;
        }

        energy /= 12f;
        jerk /= 12f;

        AddReward(-energy * actionEnergyScale);
        if (_stepId > jerkGraceSteps) AddReward(-jerk * actionJerkScale);

        distToTarget = currentDistanceToTarget;
        _prevDistance = currentDistanceToTarget;
    }

    public void FixedUpdate()
    {
        body.AddForce((cube.transform.position - body.transform.position).normalized * strenghtMove);
        for (int i = 0; i < 12; i++)
        {
            legs[i].AddForce((cube.transform.position - body.transform.position).normalized * strenghtMove / 20f);
        }

        RaycastHit hit;
        if (Physics.Raycast(body.transform.position, body.transform.right, out hit))
        {
            if (hit.collider.gameObject == cube)
            {
                body.AddForce(2f * strenghtMove * (cube.transform.position - body.transform.position).normalized);
                for (int i = 0; i < 12; i++)
                {
                    legs[i].AddForce((cube.transform.position - body.transform.position).normalized * strenghtMove / 10f);
                }
            }
        }
        Debug.DrawRay(body.transform.position, body.transform.right, Color.white);
    }

    void MoveLeg(ArticulationBody leg, float targetAngle)
    {
        leg.GetComponent<Leg>().MoveLeg(targetAngle, servoSpeed);
    }

    private float ComputeAbsHeadingDeg()
    {
        Vector3 toTarget = cube.transform.position - body.transform.position;
        if (toTarget.sqrMagnitude < 1e-6f) return 0f;
        return AbsSignedAngleDeg(body.transform.right, toTarget.normalized, Vector3.up);
    }

    private static float AbsSignedAngleDeg(Vector3 from, Vector3 to, Vector3 axis)
    {
        float a = Vector3.SignedAngle(from, to, axis);
        return Mathf.Abs(a);
    }
}
