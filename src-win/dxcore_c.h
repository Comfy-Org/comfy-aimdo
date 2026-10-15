#ifndef _DXCORE_MINIMAL_C_H_
#define _DXCORE_MINIMAL_C_H_

#include <windows.h>
#include <stdbool.h>
#include <stdint.h>
#include <initguid.h>

#ifdef __cplusplus
extern "C" {
#endif

DEFINE_GUID(IID_IDXCoreAdapterFactory, 0x78ee5945, 0xc36e, 0x4b13, 0xa6, 0x69, 0x00, 0x5d, 0xd1, 0x1c, 0x0f, 0x06);
DEFINE_GUID(IID_IDXCoreAdapter, 0xf0db4c7f, 0xfe5a, 0x42a2, 0xbd, 0x62, 0xf2, 0xa6, 0xcf, 0x6f, 0xc8, 0x3e);

typedef enum DXCoreAdapterProperty {
    DXCoreAdapterProperty_DriverDescription = 2
} DXCoreAdapterProperty;

typedef enum DXCoreAdapterState {
    DXCoreAdapterState_AdapterMemoryBudget = 1
} DXCoreAdapterState;

typedef enum DXCoreSegmentGroup {
    DXCoreSegmentGroup_Local = 0,
    DXCoreSegmentGroup_NonLocal = 1
} DXCoreSegmentGroup;

typedef struct DXCoreAdapterMemoryBudgetNodeSegmentGroup {
    uint32_t nodeIndex;
    DXCoreSegmentGroup segmentGroup;
} DXCoreAdapterMemoryBudgetNodeSegmentGroup;

typedef struct DXCoreAdapterMemoryBudget {
    uint64_t budget;
    uint64_t currentUsage;
    uint64_t availableForReservation;
    uint64_t currentReservation;
} DXCoreAdapterMemoryBudget;

typedef struct IDXCoreAdapter IDXCoreAdapter;
typedef struct IDXCoreAdapterFactory IDXCoreAdapterFactory;

/* --- IDXCoreAdapter --- */
typedef struct IDXCoreAdapterVtbl {
    // IUnknown methods
    HRESULT(STDMETHODCALLTYPE* QueryInterface)(IDXCoreAdapter* This, REFIID riid, void** ppvObject);
    ULONG(STDMETHODCALLTYPE* AddRef)(IDXCoreAdapter* This);
    ULONG(STDMETHODCALLTYPE* Release)(IDXCoreAdapter* This);

    // IDXCoreAdapter methods
    bool(STDMETHODCALLTYPE* IsValid)(IDXCoreAdapter* This);
    bool(STDMETHODCALLTYPE* IsAttributeSupported)(IDXCoreAdapter* This, REFGUID attributeGUID);
    bool(STDMETHODCALLTYPE* IsPropertySupported)(IDXCoreAdapter* This, DXCoreAdapterProperty property);
    HRESULT(STDMETHODCALLTYPE* GetProperty)(IDXCoreAdapter* This, DXCoreAdapterProperty property, size_t bufferSize, void* propertyData);
    HRESULT(STDMETHODCALLTYPE* GetPropertySize)(IDXCoreAdapter* This, DXCoreAdapterProperty property, size_t* bufferSize);
    bool(STDMETHODCALLTYPE* IsQueryStateSupported)(IDXCoreAdapter* This, DXCoreAdapterState property);
    HRESULT(STDMETHODCALLTYPE* QueryState)(IDXCoreAdapter* This, DXCoreAdapterState state, size_t inputStateDetailsSize, const void* inputStateDetails, size_t outputBufferSize, void* outputBuffer);
    bool(STDMETHODCALLTYPE* IsSetStateSupported)(IDXCoreAdapter* This, DXCoreAdapterState property);
    HRESULT(STDMETHODCALLTYPE* SetState)(IDXCoreAdapter* This, DXCoreAdapterState state, size_t inputStateDetailsSize, const void* inputStateDetails, size_t inputDataSize, const void* inputData);
    HRESULT(STDMETHODCALLTYPE* GetFactory)(IDXCoreAdapter* This, REFIID riid, void** ppvFactory);
} IDXCoreAdapterVtbl;

struct IDXCoreAdapter { IDXCoreAdapterVtbl* lpVtbl; };

/* --- IDXCoreAdapterFactory --- */
typedef struct IDXCoreAdapterFactoryVtbl {
    // IUnknown
    HRESULT(STDMETHODCALLTYPE* QueryInterface)(IDXCoreAdapterFactory* This, REFIID riid, void** ppvObject);
    ULONG(STDMETHODCALLTYPE* AddRef)(IDXCoreAdapterFactory* This);
    ULONG(STDMETHODCALLTYPE* Release)(IDXCoreAdapterFactory* This);

    // IDXCoreAdapterFactory
    HRESULT(STDMETHODCALLTYPE* CreateAdapterList)(IDXCoreAdapterFactory* This, uint32_t numAttributes, const GUID* filterAttributes, REFIID riid, void** ppvAdapterList);
    HRESULT(STDMETHODCALLTYPE* GetAdapterByLuid)(IDXCoreAdapterFactory* This, const LUID* adapterLUID, REFIID riid, void** ppvAdapter);
    bool(STDMETHODCALLTYPE* IsNotificationTypeSupported)(IDXCoreAdapterFactory* This, uint32_t notificationType);
    HRESULT(STDMETHODCALLTYPE* RegisterEventNotification)(IDXCoreAdapterFactory* This, IUnknown* dxCoreObject, uint32_t notificationType, void* callbackFunction, void* callbackContext, uint32_t* eventCookie);
    HRESULT(STDMETHODCALLTYPE* UnregisterEventNotification)(IDXCoreAdapterFactory* This, uint32_t eventCookie);
} IDXCoreAdapterFactoryVtbl;

struct IDXCoreAdapterFactory { IDXCoreAdapterFactoryVtbl* lpVtbl; };

STDAPI DXCoreCreateAdapterFactory(REFIID riid, void** ppvFactory);

#ifdef __cplusplus
}
#endif

#endif // _DXCORE_MINIMAL_C_H_
