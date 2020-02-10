import subprocess


def run(fname):
    # try:
        # Form auto-complete for recon
        out = subprocess.Popen(['tomopy', 'recon', '-h'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        stdout, _ = out.communicate()
        stdout = str(stdout)

        cmdscan = []
        parscan = []
        st = stdout.find("optional")
        while(1):
            st = stdout.find('--', st)
            end = stdout.find(' ', st)
            if(st < 0):
                break
            cmdscan.append(stdout[st:end])
            st = stdout.find('(default: ', end)
            end = stdout.find(')', st)
            parscan.append(stdout[st+10:end])
            st = end

        # Create bash file
        fid = open(fname, 'w')
        fid.write(
            '#/usr/bin/env bash\n _tomopy()\n{\n\tlocal cur prev opts\n\tCOMPREPLY=()\n\tcur="${COMP_WORDS[COMP_CWORD]}"\n\tprev="${COMP_WORDS[COMP_CWORD-1]}"\n')

        # check all tomopy recon
        fid.write('\tif [[ ${prev} == "recon" ]] ; then\n')
        fid.write('\t\topts="')
        fid.write(' '.join(cmdscan))
        fid.write('"\n')
        fid.write(
            '\t\tCOMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )\n\t\treturn 0\n\tfi\n')

        for k in range(len(cmdscan)):
            fid.write('\tif [[ ${prev} == "')
            fid.write(cmdscan[k])
            fid.write('" ]] ; then\n')
            fid.write('\t\topts="')
            fid.write(parscan[k])
            fid.write('"\n')
            fid.write(
                '\t\tCOMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )\n\t\treturn 0\n\tfi\n')
        fid.write('}\n')
        fid.write('complete -F _tomopy tomopy')
        fid.close()
    # except:
        # return 1
        return 0
