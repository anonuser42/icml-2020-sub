main() {

git filter-branch --env-filter '
export GIT_AUTHOR_EMAIL="anonymous"
export GIT_AUTHOR_NAME="anonymous"
export GIT_COMMITTER_EMAIL="anonymous"
export GIT_COMMITTER_NAME="anonymous"
' -- --all

}

main
