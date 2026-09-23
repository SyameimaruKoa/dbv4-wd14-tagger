# Bash completion for run_tagger.sh
_dbv4_wd14_tagger_complete() {
    local cur prev opts profiles split_modes
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    opts="--provider --webgpu --gpu-index --directml-device-index --webgpu-device-index --target-vendor --openvino-device --tensorrt-lib-dir -p --path -g --gpu -I --force-intel -N --force-nvidia -A --force-amd -o --organize -t --tag -x --pixiv -R --no-report -r --recursive -n --no-recursive -b --batch-size -w --io-workers -m --model-profile -e --model-repo -M --model-file -T --tags-file -q --thresh -f --force -s --sensitive-split-mode -c --record-ratio -C --no-record-ratio -S --server -K --client -L --login -H --host -P --port -d --debug -h --help"
    profiles="compact_manual lightweight medium_manual balanced high ultra wd14_v3 future_1b"
    split_modes="2 4 6"

    case "$prev" in
        --provider)
            COMPREPLY=( $(compgen -W "cpu cuda tensorrt intel webgpu migraphx" -- "$cur") )
            return ;;
        --target-vendor)
            COMPREPLY=( $(compgen -W "nvidia intel amd" -- "$cur") )
            return ;;
        --tensorrt-lib-dir)
            COMPREPLY=( $(compgen -d -- "$cur") )
            return ;;
        --gpu-index|--directml-device-index|--webgpu-device-index|--openvino-device)
            return ;;
        -p|--path)
            COMPREPLY=( $(compgen -f -- "$cur") )
            compopt -o filenames 2>/dev/null
            return
            ;;
        -M|--model-file|-T|--tags-file)
            COMPREPLY=( $(compgen -f -- "$cur") )
            compopt -o filenames 2>/dev/null
            return
            ;;
        -m|--model-profile)
            COMPREPLY=( $(compgen -W "$profiles" -- "$cur") )
            return
            ;;
        -s|--sensitive-split-mode)
            COMPREPLY=( $(compgen -W "$split_modes" -- "$cur") )
            return
            ;;
        -b|--batch-size|-w|--io-workers|-e|--model-repo|-q|--thresh|-H|--host|-P|--port)
            return
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
    else
        COMPREPLY=( $(compgen -f -- "$cur") )
        compopt -o filenames 2>/dev/null
    fi
}

complete -F _dbv4_wd14_tagger_complete run_tagger.sh
complete -F _dbv4_wd14_tagger_complete ./run_tagger.sh
