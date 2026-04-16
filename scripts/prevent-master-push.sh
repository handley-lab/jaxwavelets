#!/bin/bash
# Prevent direct pushes to master/main branch

while read local_ref local_sha remote_ref remote_sha; do
    if [ "$remote_ref" = "refs/heads/master" ] || [ "$remote_ref" = "refs/heads/main" ]; then
        echo "Direct pushes to '$remote_ref' are not allowed!"
        echo "Please use a pull request instead."
        exit 1
    fi
done

exit 0
