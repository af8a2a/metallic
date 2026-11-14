cmake_minimum_required(VERSION 3.20)

if(NOT EXISTS "${METALLIC_CTL_EXECUTABLE}")
    message(FATAL_ERROR "Build MetallicCtl before running its argument tests")
endif()

set(case_count 0)
set(failures "")
function(check_arguments expected option value)
    # --help is reached only after the preceding arguments parse successfully.
    # No debug service, named pipe, capture files or GPU are used by these cases.
    execute_process(COMMAND "${METALLIC_CTL_EXECUTABLE}" --json "${option}" "${value}" --help
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 5)
    if(expected STREQUAL "valid")
        if(NOT result STREQUAL "0" OR NOT output MATCHES "^metallicctl " OR NOT error STREQUAL "")
            list(APPEND failures "valid ${option} '${value}': exit=${result}, stdout=${output}, stderr=${error}")
        endif()
    else()
        string(JSON status ERROR_VARIABLE json_error GET "${output}" status)
        if(json_error)
            list(APPEND failures "invalid ${option} '${value}': no JSON error, exit=${result}, stdout=${output}, stderr=${error}")
        else()
            string(JSON code GET "${output}" error code)
            if(NOT result STREQUAL "1" OR NOT status STREQUAL "error" OR NOT code STREQUAL "InvalidArgument" OR NOT error STREQUAL "")
                list(APPEND failures "invalid ${option} '${value}': exit=${result}, stdout=${output}, stderr=${error}")
            endif()
        endif()
    endif()
    math(EXPR case_count "${case_count} + 1")
    set(case_count "${case_count}" PARENT_SCOPE)
    set(failures "${failures}" PARENT_SCOPE)
endfunction()

foreach(value IN ITEMS "0,0" "1,2" "1 2" " 1, 2 " "4294967295,4294967295")
    check_arguments(valid --pixel "${value}")
endforeach()
foreach(value IN ITEMS "0,0,1,1" "1 2 3 4" " 1, 2, 3, 4 " "4294967295,4294967295,4294967295,4294967295")
    check_arguments(valid --roi "${value}")
endforeach()
foreach(value IN ITEMS "" "0" "-1,0" "0,-1" "+1,0" "4294967296,0" "0,4294967296"
        "18446744073709551616,0" "0,0junk" "0,0,junk" "0,0,1" "0,0 1" "0,1.5")
    check_arguments(invalid --pixel "${value}")
endforeach()
foreach(value IN ITEMS "" "0,0" "0,0,1" "-1,0,1,1" "0,-1,1,1" "0,0,-1,1" "0,0,1,-1"
        "0,0,+1,1" "0,0,4294967296,1" "0,0,1,4294967296" "0,0,1,1junk" "0,0,1,1,junk"
        "0,0,1,1,2" "0,0,1,1 2")
    check_arguments(invalid --roi "${value}")
endforeach()

# Existing unsigned options keep their own uint32/uint64/budget boundaries.
check_arguments(valid --pid "4294967295")
check_arguments(invalid --pid "4294967296")
check_arguments(valid --count "18446744073709551615")
check_arguments(invalid --count "18446744073709551616")
check_arguments(invalid --count "-1")
check_arguments(valid --timeout-ms "300000")
check_arguments(invalid --timeout-ms "300001")

if(failures)
    list(LENGTH failures failure_count)
    message(FATAL_ERROR "${failure_count}/${case_count} CLI argument cases failed: ${failures}")
endif()
message(STATUS "${case_count} CLI argument cases passed (CPU only, no service/GPU)")
