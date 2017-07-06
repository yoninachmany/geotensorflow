BASE := $(subst -, ,$(notdir ${CURDIR}))
ORG  := $(word 1, ${BASE})
REPO := $(word 2, ${BASE})
IMG  := quay.io/${ORG}/${REPO}
TAG  := tfjava

build:
	docker build -t ${TAG}	.

# publish: build
# 	docker push ${IMG}:latest
# 	if [ "${TAG}" != "" -a "${TAG}" != "latest" ]; then docker tag ${IMG}:latest ${IMG}:${TAG} && docker push ${IMG}:${TAG}; fi

test: build
	docker run -it --rm ${TAG} java -version
