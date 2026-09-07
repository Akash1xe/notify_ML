# Notify v1 Release Checklist

- [ ] Version is `1.0.0` across release-facing metadata
- [ ] Environment/config validation passes
- [ ] Backend tests pass
- [ ] Python compileall passes
- [ ] Frontend `npm ci`, typecheck, lint, tests and build pass
- [ ] Evaluation dataset validation passes
- [ ] Locked quality baseline/regression gate passes
- [ ] Performance smoke passes
- [ ] Stress CI tier passes
- [ ] Reliability CI tier passes
- [ ] End-to-end deterministic fixture passes
- [ ] PDF integrity/download path remains valid
- [ ] `.env.example` and frontend env example are current
- [ ] No secrets, model weights, job storage, node_modules or generated metadata are committed
- [ ] Architecture/configuration/diagnostics/evaluation docs are current
- [ ] Known limitations are documented
- [ ] Official release report is `READY` or explicitly accepted `READY_WITH_WARNINGS`
- [ ] Normal main CI is green before creating `v1.0.0`
