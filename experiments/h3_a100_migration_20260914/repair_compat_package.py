"""Resume installed dependencies after fixing official Ubuntu package-name parsing."""
import json
import subprocess
import sys
import resume_dual_source as resume

resume.HISTORY = resume.WORK / 'history/compat_package_parser_20260914'
resume.PREFLIGHT_LOG = resume.WORK / 'compat_repair_preflight.log'

if sys.argv[1:] == ['--launch']:
    with (resume.WORK / 'compat_repair.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', __file__], stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({'repair_pid': child.pid}))
else:
    try:
        resume.main()
    except BaseException as exc:
        resume.record('failed', error=repr(exc))
        raise
